# nomi
fine tune LLM to play chess

# for conda setup
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -n chess-rl python=3.10
conda activate chess-rl

# for downloading llama
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets
pip install trl peft bitsandbytes
pip install python-chess


# or for downloading llama
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# authen huggingface
huggingface-cli login

# stcokfish install
apt-get update && apt-get install -y stockfish 
find /usr -name "stockfish" 2>/dev/null
export PATH="/usr/games:$PATH" (or the paths found in the previous step)
echo -e "position startpos\neval\nquit" | stockfish

# wandb login
wandb login

# run command
python train.py 2>&1 | tee out.logs

# Example data
{"fen": "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -", "depth": 39, "evaluation": "0.58", "best_move": "Kg7", "best_line": "Kg7 Re2 Rd8 Rd2 b5 Be6 Kf6 Bb3 a5 a3"}
{"fen": "8/4r3/2R2pk1/6pp/3P4/6P1/5K1P/8 b - -", "depth": 58, "evaluation": "0.00", "best_move": "Ra7", "best_line": "Ra7 Ke3 Ra3+ Ke4 Ra2 h4 gxh4 gxh4 Rh2 Rc1"}
{"fen": "6k1/6p1/8/4K3/4NN2/8/8/8 w - -", "depth": 87, "evaluation": "M18", "best_move": "Nd6", "best_line": "Nd6 Kh7 Kf5 g5 Nh5 Kh6 Ng3 Kg7 Ke6 Kf8"}
{"fen": "r1b2rk1/1p2bppp/p1nppn2/q7/2P1P3/N1N5/PP2BPPP/R1BQ1RK1 w - -", "depth": 25, "evaluation": "0.24", "best_move": "Be3", "best_line": "Be3 Rd8 Rc1 d5 cxd5 exd5 exd5 Bxa3 bxa3 Nxd5"}
{"fen": "6k1/4Rppp/8/8/8/8/5PPP/6K1 w - -", "depth": 99, "evaluation": "M1", "best_move": "Re8#", "best_line": "Re8#"}
{"fen": "6k1/6p1/6N1/4K3/4N3/8/8/8 b - -", "depth": 62, "evaluation": "M27", "best_move": "Kh7", "best_line": "Kh7 Kf5 Kh6 Ng3 Kh7 Kg5 Kg8 Ne4 Kf7 Kf5"}
{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", "depth": 64, "evaluation": "0.19", "best_move": "Nf3", "best_line": "Nf3 d5 d4 e6 c4 Nf6 Nc3 Bb4 cxd5 exd5"}
{"fen": "8/8/2N2k2/8/1p2p3/p7/K7/8 b - -", "depth": 54, "evaluation": "0.00", "best_move": "b3+", "best_line": "b3+ Kxa3 Ke6 Kxb3 Kd5 Kc3 Kxc6 Kd4 e3 Kxe3"}
{"fen": "8/1r6/2R2pk1/6pp/3P4/6P1/5K1P/8 w - -", "depth": 46, "evaluation": "0.00", "best_move": "Rc2", "best_line": "Rc2 Kf5 h4 g4 Rc5+ Ke4 Rxh5 Rb2+ Kg1 f5"}
{"fen": "1R4k1/3q1pp1/6n1/b2p2Pp/2pP2b1/p1P5/P1BQrPPB/5NK1 b - -", "depth": 31, "evaluation": "-1.05", "best_move": "Kh7", "best_line": "Kh7 Qc1 Bc7 Bxc7 Qxc7 Ra8 Qe7 f3 Bd7 Ra7"}