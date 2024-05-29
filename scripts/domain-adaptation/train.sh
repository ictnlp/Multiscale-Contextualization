TOTAL_NUM_UPDATES=500000
WARMUP_UPDATES=4000
LR=7e-04
MAX_TOKENS=16384
UPDATE_FREQ=1
datapath=Path/To/domain-adapt/wmt-byte-bin/de-en-byte
savedir=Path/To/Checkpoint_file/domain-adaptation

mkdir -p $savedir

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup fairseq-train $datapath \
	--max-tokens $MAX_TOKENS \
	--task translation \
	--truncate-source --share-all-embeddings \
	--ddp-backend=legacy_ddp \
	--share-decoder-input-output-embed \
	--required-batch-size-multiple 1 \
	--arch multiscale_transformer --criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 \
	--dropout 0.1 --conv-kernels "0 0 1 1 3 3 5 5" \
	--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
	--save-interval 1 --keep-interval-updates 20 --save-interval-updates 5000 \
	--seed 222 --patience 10 \
	--log-format simple --log-interval 100 \
	--clip-norm 0.0 \
	--weight-decay 0.0001 \
	--lr-scheduler inverse_sqrt --lr $LR \
	--max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
	--fp16 --update-freq $UPDATE_FREQ \
	--skip-invalid-size-inputs-valid-test \
	--valid-subset valid \
	--eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-print-samples \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
	--save-dir $savedir | tee -a $savedir/log.out &
