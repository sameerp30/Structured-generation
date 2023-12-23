from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=2, experiment="msmarco")):

        config = ColBERTConfig(
            bsize=16,
            root="/raid/nlp/sameer/ColBERT/checkpoints",
            checkpoint="/raid/nlp/sameer/ColBERT/colbertv2.0",
            maxsteps=18000
        )
        trainer = Trainer(
            triples="/raid/nlp/sameer/ColBERT/colbert_triples_1_pos_3_neg.json",
            queries="/raid/nlp/sameer/ColBERT/colbert_tsv_files/queries_train.tsv",
            collection="/raid/nlp/sameer/ColBERT/colbert_tsv_files/index.tsv",
            config=config,
        )

        checkpoint_path = trainer.train()

        print(f"Saved checkpoint to {checkpoint_path}...")