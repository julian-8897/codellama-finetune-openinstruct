import logging

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name="codellama/CodeLlama-7b-hf", use_qlora=True):
    """Load CodeLlama model and tokenizer with optional QLoRA"""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with QLoRA settings
    if use_qlora:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, tokenizer


def setup_peft(model, use_qlora=True):
    """Setup PEFT configuration"""

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Low rank dimension
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    if use_qlora:
        model = get_peft_model(model, peft_config)
    else:
        model = get_peft_model(model, peft_config)

    return model


def prepare_dataset(tokenizer, max_length=512):
    """Load and prepare OpenInstruct dataset"""

    # Load dataset
    dataset = load_dataset("allenai/open_instruct", "default")

    def format_instruction(example):
        """Format the instruction-response pairs"""
        instruction = example.get("instruction", "")
        response = example.get("response", "")

        # Create a simple instruction format
        formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        return {"text": formatted_text}

    def tokenize_function(examples):
        """Tokenize the examples"""
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_overflowing_tokens=False,
        )
        return outputs

    # Format and tokenize dataset
    formatted_dataset = dataset.map(
        format_instruction, remove_columns=dataset["train"].column_names
    )
    tokenized_dataset = formatted_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    return tokenized_dataset


def main():
    # Configuration
    model_name = "codellama/CodeLlama-7b-hf"
    use_qlora = True
    max_length = 512
    output_dir = "./codellama-openinstruct-lora"

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_name, use_qlora)

    logger.info("Setting up PEFT...")
    model = setup_peft(model, use_qlora)

    logger.info("Preparing dataset...")
    dataset = prepare_dataset(tokenizer, max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        save_steps=500,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb",
        run_name="codellama-openinstruct-run",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model...")
    trainer.save_model()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
