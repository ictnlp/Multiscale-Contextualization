# notice: please make sure you have `mosesdecoder` installed,
# this script will use the `detokenizer.perl` file in `mosesdecoder`.
modelfile=Path/To/Checkpoint_file/domain-adaptation
python ~/fairseq-master/scripts/average_checkpoints.py --inputs $modelfile/ --num-epoch-checkpoints 5 --output $modelfile/checkpoint_aver.pt

echo "news domain"
path_2_data=Path/To/domain-adapt/wmt-byte-bin/de-en-byte
output_dir=Path/To/Inference_file/domain-adaptation/news
mkdir -p $output_dir
mkdir -p $output_dir/raw
mkdir -p $output_dir/tmp
mkdir -p $output_dir/bleus

gen(){
model=$modelfile/checkpoint$3.pt
CUDA_VISIBLE_DEVICES=$1 fairseq-generate $path_2_data \
  --path $model \
  --beam 4 \
  --gen-subset test \
  --batch-size 128 \
  --skip-invalid-size-inputs-valid-test \
  --log-format json > $output_dir/raw/pred.$2.$3.txt

cat $output_dir/raw/pred.$2.$3.txt | grep -P "^H" |sort -V |cut -f 3- |cat > $output_dir/tmp/$2.$3.hyp 
cat $output_dir/raw/pred.$2.$3.txt | grep -P "^T" |sort -V |cut -f 2- |cat > $output_dir/tmp/$2.$3.ref

python ~/Multiscale-Attention/tools/encode-byte.py $output_dir/tmp/$2.$3.hyp $output_dir/tmp/$2.$3.str.hyp
python ~/Multiscale-Attention/tools/encode-byte.py $output_dir/tmp/$2.$3.ref $output_dir/tmp/$2.$3.str.ref

perl ~/mosesdecoder-master/scripts/tokenizer/detokenizer.perl < $output_dir/tmp/$2.$3.str.hyp > $output_dir/tmp/$2.$3.hyp.detok
perl ~/mosesdecoder-master/scripts/tokenizer/detokenizer.perl < $output_dir/tmp/$2.$3.str.ref > $output_dir/tmp/$2.$3.ref.detok

sacrebleu $output_dir/tmp/$2.$3.ref.detok --metrics bleu -w 3 --tokenize 13a < $output_dir/tmp/$2.$3.hyp.detok > $output_dir/bleus/$2.$3.bleu
}

gen 0 0 _best &
gen 1 0 _aver &
gen 2 0 _last &
wait
echo "finish news"

echo "it domain"
path_2_data=Path/To/domain-adapt/wmt-byte-bin/it-de-en-byte
output_dir=Path/To/Inference_file/domain-adaptation/it
mkdir -p $output_dir
mkdir -p $output_dir/raw
mkdir -p $output_dir/tmp
mkdir -p $output_dir/bleus

gen(){
model=$modelfile/checkpoint$3.pt
CUDA_VISIBLE_DEVICES=$1 fairseq-generate $path_2_data \
  --path $model \
  --beam 4 \
  --gen-subset test \
  --batch-size 128 \
  --skip-invalid-size-inputs-valid-test \
  --log-format json > $output_dir/raw/pred.$2.$3.txt

cat $output_dir/raw/pred.$2.$3.txt | grep -P "^H" |sort -V |cut -f 3- |cat > $output_dir/tmp/$2.$3.hyp 
cat $output_dir/raw/pred.$2.$3.txt | grep -P "^T" |sort -V |cut -f 2- |cat > $output_dir/tmp/$2.$3.ref

python ~/Multiscale-Attention/tools/encode-byte.py $output_dir/tmp/$2.$3.hyp $output_dir/tmp/$2.$3.str.hyp
python ~/Multiscale-Attention/tools/encode-byte.py $output_dir/tmp/$2.$3.ref $output_dir/tmp/$2.$3.str.ref

perl ~/mosesdecoder-master/scripts/tokenizer/detokenizer.perl < $output_dir/tmp/$2.$3.str.hyp > $output_dir/tmp/$2.$3.hyp.detok
perl ~/mosesdecoder-master/scripts/tokenizer/detokenizer.perl < $output_dir/tmp/$2.$3.str.ref > $output_dir/tmp/$2.$3.ref.detok

sacrebleu $output_dir/tmp/$2.$3.ref.detok --metrics bleu -w 3 --tokenize 13a < $output_dir/tmp/$2.$3.hyp.detok > $output_dir/bleus/$2.$3.bleu
}

gen 0 0 _best &
gen 1 0 _aver &
gen 2 0 _last &
wait
echo "finish it"

echo "koran domain"
path_2_data=Path/To/domain-adapt/wmt-byte-bin/koran-de-en-byte
output_dir=Path/To/Inference_file/domain-adaptation/koran
mkdir -p $output_dir
mkdir -p $output_dir/raw
mkdir -p $output_dir/tmp
mkdir -p $output_dir/bleus

gen(){
model=$modelfile/checkpoint$3.pt
CUDA_VISIBLE_DEVICES=$1 fairseq-generate $path_2_data \
  --path $model \
  --beam 4 \
  --gen-subset test \
  --batch-size 128 \
  --skip-invalid-size-inputs-valid-test \
  --log-format json > $output_dir/raw/pred.$2.$3.txt

cat $output_dir/raw/pred.$2.$3.txt | grep -P "^H" |sort -V |cut -f 3- |cat > $output_dir/tmp/$2.$3.hyp 
cat $output_dir/raw/pred.$2.$3.txt | grep -P "^T" |sort -V |cut -f 2- |cat > $output_dir/tmp/$2.$3.ref

python ~/Multiscale-Attention/tools/encode-byte.py $output_dir/tmp/$2.$3.hyp $output_dir/tmp/$2.$3.str.hyp
python ~/Multiscale-Attention/tools/encode-byte.py $output_dir/tmp/$2.$3.ref $output_dir/tmp/$2.$3.str.ref

perl ~/mosesdecoder-master/scripts/tokenizer/detokenizer.perl < $output_dir/tmp/$2.$3.str.hyp > $output_dir/tmp/$2.$3.hyp.detok
perl ~/mosesdecoder-master/scripts/tokenizer/detokenizer.perl < $output_dir/tmp/$2.$3.str.ref > $output_dir/tmp/$2.$3.ref.detok

sacrebleu $output_dir/tmp/$2.$3.ref.detok --metrics bleu -w 3 --tokenize 13a < $output_dir/tmp/$2.$3.hyp.detok > $output_dir/bleus/$2.$3.bleu
}

gen 0 0 _best &
gen 1 0 _aver &
gen 2 0 _last &
wait
echo "finish koran"

echo "medical domain"
path_2_data=Path/To/domain-adapt/wmt-byte-bin/medical-de-en-byte
output_dir=Path/To/Inference_file/domain-adaptation/medical
mkdir -p $output_dir
mkdir -p $output_dir/raw
mkdir -p $output_dir/tmp
mkdir -p $output_dir/bleus

gen(){
model=$modelfile/checkpoint$3.pt
CUDA_VISIBLE_DEVICES=$1 fairseq-generate $path_2_data \
  --path $model \
  --beam 4 \
  --gen-subset test \
  --batch-size 128 \
  --skip-invalid-size-inputs-valid-test \
  --log-format json > $output_dir/raw/pred.$2.$3.txt

cat $output_dir/raw/pred.$2.$3.txt | grep -P "^H" |sort -V |cut -f 3- |cat > $output_dir/tmp/$2.$3.hyp 
cat $output_dir/raw/pred.$2.$3.txt | grep -P "^T" |sort -V |cut -f 2- |cat > $output_dir/tmp/$2.$3.ref

python ~/Multiscale-Attention/tools/encode-byte.py $output_dir/tmp/$2.$3.hyp $output_dir/tmp/$2.$3.str.hyp
python ~/Multiscale-Attention/tools/encode-byte.py $output_dir/tmp/$2.$3.ref $output_dir/tmp/$2.$3.str.ref

perl ~/mosesdecoder-master/scripts/tokenizer/detokenizer.perl < $output_dir/tmp/$2.$3.str.hyp > $output_dir/tmp/$2.$3.hyp.detok
perl ~/mosesdecoder-master/scripts/tokenizer/detokenizer.perl < $output_dir/tmp/$2.$3.str.ref > $output_dir/tmp/$2.$3.ref.detok

sacrebleu $output_dir/tmp/$2.$3.ref.detok --metrics bleu -w 3 --tokenize 13a < $output_dir/tmp/$2.$3.hyp.detok > $output_dir/bleus/$2.$3.bleu
}

gen 0 0 _best &
gen 1 0 _aver &
gen 2 0 _last &
wait
echo "finish medical"

echo "wmt19biomedical domain"
path_2_data=Path/To/domain-adapt/wmt-byte-bin/wmt19biomedical-de-en-byte
output_dir=Path/To/Inference_file/domain-adaptation/wmt19biomedical
mkdir -p $output_dir
mkdir -p $output_dir/raw
mkdir -p $output_dir/tmp
mkdir -p $output_dir/bleus

gen(){
model=$modelfile/checkpoint$3.pt
CUDA_VISIBLE_DEVICES=$1 fairseq-generate $path_2_data \
  --path $model \
  --beam 4 \
  --gen-subset test \
  --batch-size 128 \
  --skip-invalid-size-inputs-valid-test \
  --log-format json > $output_dir/raw/pred.$2.$3.txt

cat $output_dir/raw/pred.$2.$3.txt | grep -P "^H" |sort -V |cut -f 3- |cat > $output_dir/tmp/$2.$3.hyp 
cat $output_dir/raw/pred.$2.$3.txt | grep -P "^T" |sort -V |cut -f 2- |cat > $output_dir/tmp/$2.$3.ref

python ~/Multiscale-Attention/tools/encode-byte.py $output_dir/tmp/$2.$3.hyp $output_dir/tmp/$2.$3.str.hyp
python ~/Multiscale-Attention/tools/encode-byte.py $output_dir/tmp/$2.$3.ref $output_dir/tmp/$2.$3.str.ref

perl ~/mosesdecoder-master/scripts/tokenizer/detokenizer.perl < $output_dir/tmp/$2.$3.str.hyp > $output_dir/tmp/$2.$3.hyp.detok
perl ~/mosesdecoder-master/scripts/tokenizer/detokenizer.perl < $output_dir/tmp/$2.$3.str.ref > $output_dir/tmp/$2.$3.ref.detok

sacrebleu $output_dir/tmp/$2.$3.ref.detok --metrics bleu -w 3 --tokenize 13a < $output_dir/tmp/$2.$3.hyp.detok > $output_dir/bleus/$2.$3.bleu
}

gen 0 0 _best &
gen 1 0 _aver &
gen 2 0 _last &
wait
echo "finish wmt19biomedical"