import torch
import torch.nn.functional as F
import lightning as L
from transformers import RobertaForMaskedLM, GPT2LMHeadModel

from utils.data import get_tokenizer
from utils.torch import calc_entropy, sample_prob, nucleus_sampling, select_pos
from utils.operations import calc_weighted_average


class FiLM(L.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.save_hyperparameters(hparams)
        hparams = self.hparams  # a["key"] (if so) -> a.key

        self.tokenizer = get_tokenizer(hparams.pretrained_model)
        if "roberta" in hparams.pretrained_model:
            self.model = RobertaForMaskedLM.from_pretrained(hparams.pretrained_model)
            self.bos_id = None
        elif "gpt2" in hparams.pretrained_model:
            self.model = GPT2LMHeadModel.from_pretrained(hparams.pretrained_model)
            for gpt2block in self.model.transformer.h:
                gpt2block.attn.bias.fill_(True)  # remove causal mask
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.bos_id = self.tokenizer.bos_token_id
        else:
            raise ValueError
        self.mask_id = self.tokenizer.mask_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = None

    def forward(self, input_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=(input_ids != self.pad_id), labels=labels)
        logits, loss = outputs.logits, outputs.loss
        if "gpt2" in self.hparams.pretrained_model:
            logits = logits.roll(shifts=1, dims=1)
        return logits, loss

    def rand_mask(self, ids, prob):
        noise = torch.rand(ids.shape, device=self.device)
        special_tokens = (ids == self.bos_id) | (ids == self.eos_id) | (ids == self.pad_id)
        mask = (noise < prob.unsqueeze(dim=1)) & (~special_tokens)
        return torch.where(mask, self.mask_id, ids)

    def compute_loss_prob(self, target_ids, prob):
        input_ids = self.rand_mask(target_ids, prob)
        if (input_ids != self.mask_id).all():
            input_ids = torch.where(target_ids != self.pad_id, self.mask_id, self.pad_id)

        mask = (input_ids == self.mask_id)
        labels = torch.where(mask, target_ids, -100)
        _, loss = self(input_ids, labels)
        return loss, mask.sum()

    def compute_loss(self, target_ids, n=1):
        res = []
        for _ in range(n):
            prob = sample_prob(len(target_ids), self.hparams.weight_func, self.hparams.weight_param).to(self.device)
            res.append(self.compute_loss_prob(target_ids, prob))
        return calc_weighted_average(res)

    @torch.no_grad()
    def evaluate(self, input_ids, target_ids, order):
        row = torch.arange(len(input_ids), device=self.device)
        loss = torch.zeros(len(input_ids), device=self.device)
        num_masks = (input_ids == self.mask_id).sum(dim=1)
        output_ids = input_ids.clone()
        while (output_ids == self.mask_id).any():
            logits, _ = self(output_ids)
            ent = calc_entropy(logits)

            mask = (output_ids == self.mask_id)
            empty = (~mask).all(dim=1)
            col = select_pos(order, mask, empty, ent)

            labels = torch.where(~empty, target_ids[row, col], -100)
            loss += F.cross_entropy(logits[row, col], labels, reduction="none")
            output_ids[row, col] = target_ids[row, col]
        loss /= num_masks.clamp(min=1)
        return loss, num_masks

    @torch.no_grad()
    def generate(self, input_ids, order, temp, top_p):
        row = torch.arange(len(input_ids), device=self.device)
        output_ids = input_ids.clone()
        while (output_ids == self.mask_id).any():
            logits, _ = self(output_ids)
            ent = calc_entropy(logits)
            pred_ids = nucleus_sampling(logits / temp, top_p)

            mask = (output_ids == self.mask_id)
            empty = (~mask).all(dim=1)
            col = select_pos(order, mask, empty, ent)

            output_ids[row, col] = torch.where(~empty, pred_ids[row, col], output_ids[row, col])
        return output_ids

    def training_step(self, batch, batch_idx):
        loss, _ = self.compute_loss(batch[0])
        self.log("loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.compute_loss(batch[0], self.hparams.valid_samples)
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        loss, _ = calc_weighted_average(self.validation_step_outputs)
        self.log("valid_loss", loss, sync_dist=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        if self.hparams_test.task == "eval":
            output = self.evaluate(batch[0][0], batch[1][0], self.hparams_test.order)
        else:
            output = self.generate(batch[0], self.hparams_test.order, self.hparams_test.temp, self.hparams_test.top_p)
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        if self.hparams_test.task == "eval":
            loss = torch.cat([output[0] for output in self.test_step_outputs])
            num_masks = torch.cat([output[1] for output in self.test_step_outputs])
            loss_masks = torch.stack((loss, num_masks), dim=1)
            self.loss_masks = loss_masks.tolist()
            self.nll, _ = calc_weighted_average(loss_masks)
        else:
            self.outputs = []
            for output in self.test_step_outputs:
                self.outputs += output.tolist()
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
