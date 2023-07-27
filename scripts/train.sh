export PYTHONUNBUFFERED=1

rm -f nohup.out

nohup python src/train.py &
