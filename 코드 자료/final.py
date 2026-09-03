"""
012 노트북에서 학습해 저장한 chatbot_transformer.pt 체크포인트를 불러와
Streamlit 채팅 UI로 실시간 대화하는 간단한 Agent.

실행: streamlit run final.py
(이 파일과 같은 폴더에 chatbot_transformer.pt가 있어야 한다)

모델 클래스는 012 노트북의 것과 동일하게 이 파일 안에 직접 담았다 (모듈로 분리하지 않음).
체크포인트(state_dict)를 불러오려면 클래스 구조가 정확히 같아야 하므로,
012 노트북에서 모델 구조를 바꾸면 이 파일도 반드시 같이 맞춰야 한다.
"""

import math
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn

CHECKPOINT_PATH = "chatbot_transformer.pt"


# ============================================================
# 0. 012 노트북과 동일한 Mask / PositionalEncoding / Attention
# ============================================================
def make_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).bool()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T]


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = weights @ V

    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, _ = x.shape
        x = x.view(B, T, self.h, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        out, _ = scaled_dot_product_attention(Q, K, V, mask)

        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.W_o(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 1. Encoder / Decoder
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, causal_mask=None, pad_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, causal_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, pad_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_out, causal_mask=None, pad_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, causal_mask, pad_mask)
        return x


# ============================================================
# 2. 전체 챗봇 Transformer
# ============================================================
class ChatbotTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=3, num_heads=8,
                 d_ff=512, max_len=64, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx

        self.embed_layer = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pe_layer = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

        self.output_linear = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        src_pad_mask = make_padding_mask(src, self.pad_idx)
        x = self.dropout(self.pe_layer(self.embed_layer(src)))
        enc_out = self.encoder(x, src_pad_mask)
        return enc_out, src_pad_mask

    def decode(self, tgt, enc_out, src_pad_mask):
        T = tgt.size(1)
        causal_mask = make_causal_mask(T).to(tgt.device)
        tgt_pad_mask = make_padding_mask(tgt, self.pad_idx)
        combined_mask = causal_mask & tgt_pad_mask

        x = self.dropout(self.pe_layer(self.embed_layer(tgt)))
        dec_out = self.decoder(x, enc_out, combined_mask, src_pad_mask)

        return self.output_linear(dec_out)

    def forward(self, src, tgt):
        enc_out, src_pad_mask = self.encode(src)
        return self.decode(tgt, enc_out, src_pad_mask)


# ============================================================
# 3. Tokenizer decode + 추론 (Greedy Decoding)
# ============================================================
def encode_text(sentence, stoi):
    ids = [stoi.get(w, stoi["<unk>"]) for w in sentence.split()]
    return [stoi["<sos>"]] + ids + [stoi["<eos>"]]


def decode_ids(ids, itos):
    specials = {"<pad>", "<sos>", "<eos>"}
    words = []
    for i in ids:
        w = itos.get(int(i), "<unk>")
        if w == "<eos>":
            break
        if w in specials:
            continue
        words.append(w)
    return " ".join(words)


def pad_ids(ids, max_len, pad_idx=0):
    ids = ids[:max_len]
    return ids + [pad_idx] * (max_len - len(ids))


def generate_reply(model, question, stoi, itos, max_len=30, device="cpu"):
    model.eval()
    src_ids = pad_ids(encode_text(question, stoi), max_len)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        enc_out, src_pad_mask = model.encode(src)

        tgt_ids = [stoi["<sos>"]]
        for _ in range(max_len):
            tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
            logits = model.decode(tgt, enc_out, src_pad_mask)
            next_id = logits[0, -1].argmax(-1).item()

            if next_id == stoi["<eos>"]:
                break
            tgt_ids.append(next_id)

    reply = decode_ids(tgt_ids[1:], itos)
    return reply if reply else "음... 잘 모르겠어요."


# ============================================================
# 4. Streamlit 앱
# ============================================================
st.set_page_config(page_title="한국어 챗봇 Agent", page_icon="💬")
st.title("나의 한국어 챗봇 Agent")
st.caption("00~05주차에서 만든 Transformer를 직접 학습시킨 모델 + 하드코딩 Function Calling")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model = ChatbotTransformer(
        vocab_size=len(checkpoint["stoi"]),
        pad_idx=checkpoint["stoi"]["<pad>"],
        **checkpoint["config"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["stoi"], checkpoint["itos"], device


try:
    model, stoi, itos, device = load_model()
except FileNotFoundError:
    st.error(
        f"체크포인트 파일을 찾을 수 없습니다: `{CHECKPOINT_PATH}`\n\n"
        "먼저 `012. final.ipynb` 노트북을 끝까지 실행해서 모델을 학습시키고 저장하세요."
    )
    st.stop()


# ============================================================
# 하드코딩 Function Calling (13주차 Agent 패턴)
# 실무에서는 이 라우팅 판단 자체를 모델에게 맡기지만(진짜 function calling),
# 여기서는 MVP로 키워드 기반 규칙만 사용한다.
# ============================================================
def get_current_time():
    return datetime.now().strftime("%H시 %M분")


def get_current_date():
    weekday = ["월", "화", "수", "목", "금", "토", "일"][datetime.now().weekday()]
    return datetime.now().strftime(f"%Y년 %m월 %d일 ({weekday}요일)")


def agent_reply(user_input):
    if "몇 시" in user_input or "시간" in user_input:
        return f"지금은 {get_current_time()}이에요."

    if "날짜" in user_input or "며칠" in user_input or "요일" in user_input:
        return f"오늘은 {get_current_date()}입니다."

    return generate_reply(model, user_input, stoi, itos, device=device)


# ============================================================
# Streamlit 채팅 UI (13주차 자료와 동일한 3종 컴포넌트)
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("메시지를 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    reply = agent_reply(user_input)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
