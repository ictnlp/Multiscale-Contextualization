modelfile=Path/To/Checkpoint_file/OPUS7
output_dir=Path/To/Inference_file/OPUS7
lang_pairs="ar_AR-en_XX,zh_CN-en_XX,nl_XX-en_XX,ru_RU-en_XX,fr_XX-en_XX,de_DE-en_XX,en_XX-ar_AR,en_XX-zh_CN,en_XX-nl_XX,en_XX-ru_RU,en_XX-fr_XX,en_XX-de_DE"
lang_list=Path/To/Langlist/ML50_langs.txt
path_2_data=Path/To/OPUS7/by-preprocess
python ~/fairseq-master/scripts/average_checkpoints.py --inputs $modelfile/ --num-epoch-checkpoints 5 --output $modelfile/checkpoint_aver-epoch.pt
python ~/fairseq-master/scripts/average_checkpoints.py --inputs $modelfile/ --num-update-checkpoints 5 --output $modelfile/checkpoint_aver-update.pt

mkdir -p $output_dir
mkdir -p $output_dir/raw
mkdir -p $output_dir/tmp
mkdir -p $output_dir/bleus

gen(){
srclang=$2
tgtlang=$4
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
}

for ckpt in _best _last _aver-epoch _aver-update
do
  gen 0 ar_AR $ckpt en_XX &
  gen 1 zh_CN $ckpt en_XX &
  gen 2 nl_XX $ckpt en_XX &
  gen 3 ru_RU $ckpt en_XX &
  wait
  gen 0 fr_XX $ckpt en_XX &
  gen 1 de_DE $ckpt en_XX &
  gen 2 en_XX $ckpt ar_AR &
  gen 3 en_XX $ckpt zh_CN &
  wait
  gen 0 en_XX $ckpt nl_XX &
  gen 1 en_XX $ckpt ru_RU &
  gen 2 en_XX $ckpt fr_XX &
  gen 3 en_XX $ckpt de_DE &
  wait
done
