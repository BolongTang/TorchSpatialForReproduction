from dataloader import *
from trainer_helper import *
from eval_helper import *
from trainer import *

def main():
    parser = make_args_parser()
    args = parser.parse_args()

    # Initialize trainer
    trainer = Trainer(args, console=True)

    # Run training
    # trainer.run_train() adds the encoder on top of the image-only (no_prior) model. Do not comment it out.
    trainer.run_train()

    # Final evaluation
    trainer.run_eval_final()

    # Spatial encoder evaluation
    val_preds = trainer.run_eval_spa_enc_only(
            eval_flag_str="LocEnc ", load_model=trainer.params["spa_enc_type"] != "no_prior")

if __name__ == "__main__":
    main()
