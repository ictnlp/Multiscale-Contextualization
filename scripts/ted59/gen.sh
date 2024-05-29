modelfile=Path/To/Checkpoint_file/Ted59
output_dir=Path/To/Inference_file/Ted59
lang_pairs="pt-en,ar-en,ms-en,th-en,lt-en,zhcn-en,da-en,hi-en,sv-en,et-en,hr-en,eo-en,he-en,ko-en,ja-en,ku-en,gl-en,es-en,mn-en,fr-en,el-en,ta-en,tr-en,sq-en,ptbr-en,ro-en,eu-en,frca-en,hy-en,ur-en,fi-en,my-en,cs-en,bg-en,mr-en,de-en,vi-en,sl-en,ka-en,sk-en,nl-en,be-en,zhtw-en,bn-en,uk-en,nb-en,az-en,bs-en,zh-en,it-en,ru-en,mk-en,sr-en,hu-en,pl-en,id-en,kk-en,fa-en"
lang_list=Path/To/Langlist/ML50_langs.txt
path_2_data=Path/To/Ted59/by-preprocess
python ~/fairseq-master/scripts/average_checkpoints.py --inputs $modelfile/ --num-epoch-checkpoints 5 --output $modelfile/checkpoint_aver-epoch.pt
python ~/fairseq-master/scripts/average_checkpoints.py --inputs $modelfile/ --num-update-checkpoints 5 --output $modelfile/checkpoint_aver-update.pt

mkdir -p $output_dir
mkdir -p $output_dir/raw
mkdir -p $output_dir/tmp
mkdir -p $output_dir/bleus

gen(){
srclang=$2
tgtlang=$4
summary=$output_dir/tot-bleus$3.txt
model=$modelfile/checkpoint$3.pt
CUDA_VISIBLE_DEVICES=$1 fairseq-generate $path_2_data \
  --path $model \
  --task translation_multi_simple_epoch \
  --gen-subset test \
  --source-lang $srclang \
  --target-lang $tgtlang \
  --batch-size 64 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lenpen 1.5 \
  --skip-invalid-size-inputs-valid-test \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" > $output_dir/raw/pred.$srclang-$tgtlang.$3.txt

cat $output_dir/raw/pred.$srclang-$tgtlang.$3.txt | grep -P "^H" |sort -V |cut -f 3- |cat > $output_dir/tmp/$srclang-$tgtlang.$3.hyp 
cat $output_dir/raw/pred.$srclang-$tgtlang.$3.txt | grep -P "^T" |sort -V |cut -f 2- |cat > $output_dir/tmp/$srclang-$tgtlang.$3.ref

sacrebleu $output_dir/tmp/$srclang-$tgtlang.$3.ref --metrics bleu -w 3 --tokenize 13a < $output_dir/tmp/$srclang-$tgtlang.$3.hyp > $output_dir/bleus/$srclang-$tgtlang.$3.bytebleu

python ~/Multiscale-Attention/tools/encode-byte.py $output_dir/tmp/$srclang-$tgtlang.$3.hyp $output_dir/tmp/$srclang-$tgtlang.$3.str.hyp
python ~/Multiscale-Attention/tools/encode-byte.py $output_dir/tmp/$srclang-$tgtlang.$3.ref $output_dir/tmp/$srclang-$tgtlang.$3.str.ref

sacrebleu $output_dir/tmp/$srclang-$tgtlang.$3.str.ref --metrics bleu -w 3 --tokenize 13a < $output_dir/tmp/$srclang-$tgtlang.$3.str.hyp > $output_dir/bleus/$srclang-$tgtlang.$3.bleu

python ~/Multiscale-Attention/tools/get-bleus.py $srclang-$tgtlang$3 $output_dir/bleus/$srclang-$tgtlang.$3.bleu >> $summary
}

for ckpt in _last _best _aver-epoch _aver-update
do
gen 0 'pt' $ckpt en &
gen 1 'ar' $ckpt en &
gen 2 'ms' $ckpt en &
gen 3 'th' $ckpt en &
wait
gen 0 'lt' $ckpt en &
gen 1 'zhcn' $ckpt en &
gen 2 'da' $ckpt en &
gen 3 'hi' $ckpt en &
wait
gen 0 'sv' $ckpt en &
gen 1 'et' $ckpt en &
gen 2 'hr' $ckpt en &
gen 3 'eo' $ckpt en &
wait
gen 0 'he' $ckpt en &
gen 1 'ko' $ckpt en &
gen 2 'ja' $ckpt en &
gen 3 'ku' $ckpt en &
wait
gen 0 'gl' $ckpt en &
gen 1 'es' $ckpt en &
gen 2 'mn' $ckpt en &
gen 3 'fr' $ckpt en &
wait
gen 0 'el' $ckpt en &
gen 1 'ta' $ckpt en &
gen 2 'tr' $ckpt en &
gen 3 'sq' $ckpt en &
wait
gen 0 'ptbr' $ckpt en &
gen 1 'ro' $ckpt en &
gen 2 'eu' $ckpt en &
gen 3 'frca' $ckpt en &
wait
gen 0 'hy' $ckpt en &
gen 1 'ur' $ckpt en &
gen 2 'fi' $ckpt en &
gen 3 'my' $ckpt en &
wait
gen 0 'cs' $ckpt en &
gen 1 'bg' $ckpt en &
gen 2 'mr' $ckpt en &
gen 3 'de' $ckpt en &
wait
gen 0 'vi' $ckpt en &
gen 1 'sl' $ckpt en &
gen 2 'ka' $ckpt en &
gen 3 'sk' $ckpt en &
wait
gen 0 'nl' $ckpt en &
gen 1 'be' $ckpt en &
gen 2 'zhtw' $ckpt en &
gen 3 'bn' $ckpt en &
wait
gen 0 'uk' $ckpt en &
gen 1 'nb' $ckpt en &
gen 2 'az' $ckpt en &
gen 3 'bs' $ckpt en &
wait
gen 0 'zh' $ckpt en &
gen 1 'it' $ckpt en &
gen 2 'ru' $ckpt en &
gen 3 'mk' $ckpt en &
wait
gen 0 'sr' $ckpt en &
gen 1 'hu' $ckpt en &
gen 2 'pl' $ckpt en &
gen 3 'id' $ckpt en &
wait
gen 0 'kk' $ckpt en &
gen 1 'fa' $ckpt en &
wait
# gen 4 'pt' $ckpt en &
# gen 5 'ar' $ckpt en &
# gen 6 'ms' $ckpt en &
# gen 7 'th' $ckpt en &
# wait
# gen 4 'lt' $ckpt en &
# gen 5 'zhcn' $ckpt en &
# gen 6 'da' $ckpt en &
# gen 7 'hi' $ckpt en &
# wait
# gen 4 'sv' $ckpt en &
# gen 5 'et' $ckpt en &
# gen 6 'hr' $ckpt en &
# gen 7 'eo' $ckpt en &
# wait
# gen 4 'he' $ckpt en &
# gen 5 'ko' $ckpt en &
# gen 6 'ja' $ckpt en &
# gen 7 'ku' $ckpt en &
# wait
# gen 4 'gl' $ckpt en &
# gen 5 'es' $ckpt en &
# gen 6 'mn' $ckpt en &
# gen 7 'fr' $ckpt en &
# wait
# gen 4 'el' $ckpt en &
# gen 5 'ta' $ckpt en &
# gen 6 'tr' $ckpt en &
# gen 7 'sq' $ckpt en &
# wait
# gen 4 'ptbr' $ckpt en &
# gen 5 'ro' $ckpt en &
# gen 6 'eu' $ckpt en &
# gen 7 'frca' $ckpt en &
# wait
# gen 4 'hy' $ckpt en &
# gen 5 'ur' $ckpt en &
# gen 6 'fi' $ckpt en &
# gen 7 'my' $ckpt en &
# wait
# gen 4 'cs' $ckpt en &
# gen 5 'bg' $ckpt en &
# gen 6 'mr' $ckpt en &
# gen 7 'de' $ckpt en &
# wait
# gen 4 'vi' $ckpt en &
# gen 5 'sl' $ckpt en &
# gen 6 'ka' $ckpt en &
# gen 7 'sk' $ckpt en &
# wait
# gen 4 'nl' $ckpt en &
# gen 5 'be' $ckpt en &
# gen 6 'zhtw' $ckpt en &
# gen 7 'bn' $ckpt en &
# wait
# gen 4 'uk' $ckpt en &
# gen 5 'nb' $ckpt en &
# gen 6 'az' $ckpt en &
# gen 7 'bs' $ckpt en &
# wait
# gen 4 'zh' $ckpt en &
# gen 5 'it' $ckpt en &
# gen 6 'ru' $ckpt en &
# gen 7 'mk' $ckpt en &
# wait
# gen 4 'sr' $ckpt en &
# gen 5 'hu' $ckpt en &
# gen 6 'pl' $ckpt en &
# gen 7 'id' $ckpt en &
# wait
# gen 4 'kk' $ckpt en &
# gen 5 'fa' $ckpt en &
wait
python ~/Multiscale-Attention/tools/avg-bleus.py $output_dir/tot-bleus$ckpt.txt >> $output_dir/tot-bleus$ckpt.txt
done
