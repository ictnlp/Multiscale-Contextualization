DATA=Path/To/Ted59/by-preprocess
lang_pairs="pt-en,ar-en,ms-en,th-en,lt-en,zhcn-en,da-en,hi-en,sv-en,et-en,hr-en,eo-en,he-en,ko-en,ja-en,ku-en,gl-en,es-en,mn-en,fr-en,el-en,ta-en,tr-en,sq-en,ptbr-en,ro-en,eu-en,frca-en,hy-en,ur-en,fi-en,my-en,cs-en,bg-en,mr-en,de-en,vi-en,sl-en,ka-en,sk-en,nl-en,be-en,zhtw-en,bn-en,uk-en,nb-en,az-en,bs-en,zh-en,it-en,ru-en,mk-en,sr-en,hu-en,pl-en,id-en,kk-en,fa-en"
lang_list=Path/To/Langlist/ML50_langs.txt
savedir=Path/To/Checkpoint_file/Ted59

mkdir -p $savedir

TOTAL_NUM_UPDATES=150000
WARMUP_UPDATES=4000
LR=5e-04
MAX_TOKENS=8192
UPDATE_FREQ=2

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup fairseq-train $DATA \
	--max-tokens $MAX_TOKENS \
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
	--conv-kernels "0 0 3 3 5 5 7 7" \
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
