DATA=Path/To/OPUS7/by-preprocess
lang_pairs="ar_AR-en_XX,zh_CN-en_XX,nl_XX-en_XX,ru_RU-en_XX,fr_XX-en_XX,de_DE-en_XX,en_XX-ar_AR,en_XX-zh_CN,en_XX-nl_XX,en_XX-ru_RU,en_XX-fr_XX,en_XX-de_DE"
lang_list=Path/To/Langlist/ML50_langs.txt
savedir=Path/To/Checkpoint_file/OPUS7

mkdir -p $savedir

TOTAL_NUM_UPDATES=1000000
WARMUP_UPDATES=4000
LR=7e-04
MAX_TOKENS=4096
UPDATE_FREQ=4

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup fairseq-train $DATA \
	--max-tokens $MAX_TOKENS \
	--reset-dataloader \
	--task translation_multi_simple_epoch \
    --sampling-method "temperature" \
    --sampling-temperature 1.5 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" \
    --max-target-positions 4096 --max-source-positions 4096 \
	--truncate-source --share-all-embeddings \
	--ddp-backend=legacy_ddp \
	--share-decoder-input-output-embed \
	--conv-kernels "0 0 1 1 3 5 5 7" \
	--required-batch-size-multiple 1 \
	--arch multiscale_transformer --criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 \
	--dropout 0.1 \
	--patience 10 \
	--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
	--save-interval 1 --keep-interval-updates 20 --save-interval-updates 5000 \
	--seed 222 \
	--log-format simple --log-interval 100 \
	--clip-norm 0.0 \
	--lr-scheduler inverse_sqrt --lr $LR \
	--max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
	--fp16 --update-freq $UPDATE_FREQ \
	--skip-invalid-size-inputs-valid-test \
	--valid-subset valid \
	--save-dir $savedir | tee -a $savedir/log.out &
	# --eval-bleu \
	# --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
	# --eval-bleu-detok moses \
	# --eval-bleu-remove-bpe \
	# --eval-bleu-print-samples \
	# --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
