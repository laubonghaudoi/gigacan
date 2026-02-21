# GigaCan

- [Qwen3 ASR](https://qwen.ai/blog?id=41e4c0f6175f9b004a03a07e42343eaaf48329e7&from=research.latest-advancements-list)
- [Aliyun api key](https://bailian.console.aliyun.com/?tab=model#/api-key)
- 模型 [qwen3-asr-flash](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/group-qwen3-asr-flash?modelGroup=group-qwen3-asr-flash)
- [阿里雲餘額](https://billing-cost.console.aliyun.com/fortune/billing-account)

## 1 下載影片
 
呢一步要反覆試錯，因為 yt-dlp 下載經常會中斷。而且需要好多儲存空間，所以要經常𥄫住然後人手重試。首先要確定目標頻道或者播放清單，然後針對

1. 準備好 GCP 嘅 YouTube API key，開一個 `.env` 放個 `YOUTUBE_API_KEY=AIxxxxx` 然後跑 `uv run 1_get_video_list.py` 將指定頻道或者播放清單入面所有影片嘅 metadata 爬落一個 csv 文件度，呢個 csv 亦用於登記下載進度。
1. `uv run 2_download_audio.py`，會按照上面嘅 csv 記錄嘅進度，將未下載嘅片下載落`download/`並轉化成 16kHz OPUS 格式。最好放一個`cookies.txt`否則youtube會反爬蟲突然中斷下載。
    1. 因為下載過程會經常因為 YouTube 反爬蟲、空間唔夠等等意外中斷，所以需要有呢個 csv 嚟記錄進度。如果下載中斷，可以跑 `uv run 2_scan_progress.py`，會自動檢查 `download/` 入面邊啲已經下載咗邊啲未下載，然後更新個 csv 將啲已經下載且轉碼成功嘅登記為 `downloaded=True`。
    1. 每次中斷後重新跑 `2_download_audio.py` 都會自動讀取個 csv，按照 `downloaded`嗰列 `false` 嘅嚟下載。
    1. 下載完之後跑個 `2_organize_downloads.py` 會自動將 `download/` 入面下載好嘅音頻按照年份分類。
    1. 全部下載完成之後，跑一次
        ```bash
        uv run 2_check_audio_integrity.py download/ legco.csv --cleanup --auto-yes
        ```
        會自動按照個 csv 檢查一次所有下載好嘅 OPUS。如果遇到個長度唔對應嘅，就會刪除呢條 OPUS 然後喺 csv 入面標記`downloaded=False`
    1. 再重複跑 `uv run 2_download_audio.py`，直至將所有 opus 都下載齊為止。

## 2 轉寫字幕

### 全量轉寫（推薦）

用 `tmux` 跑生產腳本，會自動記錄 GPU/RAM/進度：

```bash
# Qwen3-ASR 1.7B + vLLM（預設，質素最好）
tmux new -s transcribe './run_production.sh'

# SenseVoice（推理快 ~13%，但質素略遜）
tmux new -s transcribe './run_production.sh sensevoice'
```

所有參數已經針對 RTX 5090 + 9950X3D + 58 GB RAM 調校至最優，唔需要手動加任何 flag。日誌輸出喺 `benchmarks/production_<timestamp>/`。

睇進度（attach 返去 tmux session 就會見到 tqdm 進度條）：

```bash
tmux attach -t transcribe
```

按 `Ctrl+B` 然後 `D` 可以 detach 返出嚟（唔會中斷任務）。

亦可以喺另一個 terminal 睇 GPU/RAM 監控：

```bash
tail -f benchmarks/production_*/monitor.log
```

如果中斷咗，直接重新跑返就得，已轉寫嘅會自動跳過，VAD 結果亦有 cache。

### 最優預設參數

Pipeline 預設值（`src/gigacan/qwen_srt/config.py`）：

| 參數 | 預設值 | 說明 |
| ---- | -----: | ---- |
| VAD pre-compute workers | 8 | Phase 1 multiprocessing VAD 工人數 |
| `prep_workers` / `vad_workers` | 24 | Phase 2 解碼工人數 |
| `super_batch_max_decoded_gib` | 25.0 | 解碼音頻 RAM 上限 |
| `vllm_max_model_len` | 4096 | vLLM KV cache 長度（ASR 序列只需 ~300-500 token） |
| `vllm_max_num_seqs` | 256 | vLLM 並行序列數 |
| `vllm_gpu_memory_utilization` | 0.9 | vLLM GPU 記憶體佔用比例 |
| `segment_batch_size` | 1536 | segment queue 容量 |

Pipeline 會先用 multiprocessing 預計算所有 VAD（Phase 1，CPU 全核跑滿），然後再跑 GPU 推理（Phase 2，GPU 35-96%）。兩個階段都有 tqdm 進度條。詳見 `optimization.md`。

### 其他用法

1. `uv run transcribe`（預設會遞迴掃描 `download/`，並將字幕輸出到 `transcriptions/<year>/*.srt`）
1. 只跑某一年：`uv run transcribe --year 2025`
1. 預設使用 `--asr-engine qwen3`（Qwen3-ASR-1.7B + vLLM）；可用 `--asr-engine sensevoice` 切換
1. `qwen3` 可調：`--vllm-gpu-memory-utilization`、`--vllm-tensor-parallel-size`、`--qwen-language`、`--qwen-context`、`--use-prompt`
1. `sensevoice` 可調：`--asr-model-hub`、`--asr-language`、`--no-asr-use-itn`
1. 單檔模式：`uv run transcribe --audio download/2026/J-ajS2LNnfs.opus --output-srt ./no_prompt.srt`
1. 用 `zh-hk` 參考字幕修正 Qwen 轉寫（唔會用 `yue` 直接改字）：
   ```bash
   uv run 6_correct_transcriptions.py --year 2025
   ```
   如要改用 Ollama（例如 `gemma3:27b`）：
   ```bash
   uv run 6_correct_transcriptions.py --year 2025 --backend ollama --model gemma3:27b
   ```
   1. 修正後輸出：`corrected_transcriptions/<year>/*.srt`
   1. 清單報表：`logs/correction_manifest_<year>.csv`
   1. 匯總報告：`logs/correction_report_<year>.json`
   1. `yue` 差異報告（只分析不改字）：`logs/yue_drift_report_<year>.csv`
1. 針對影片類別修改 system  prompt，然後跑 `2_vtt.py`，會用 silero-vad 將輸入音頻分段再叫 qwen3-asr-flash 轉寫成粵文，生成 .vtt 字幕文件到 `vtt/`。
    1. 記得修改 `2_vtt.py` 入面嘅 prompt，會對字幕準確度有好大影響。
    1. 唔同題材需要設定唔同嘅`--vad-merge-ms`時長，例如張悦楷三國演義最優大概係 450，而毛澤東的黃昏歲月就最好係 500。推薦每加一個新題材之前用`tune_vad.ipynb`嚟確定最優值。
1. 跑 `3_cut.py` 用生成嘅字幕分割音頻，生成切割後音頻到`cut/`入面，同時生成對應 `metadata.csv`
1. 讀取 `cut/` 入面切分好嘅音頻，上傳數據集到 HuggingFace
1. 刪除 `download`/ 同埋 `vtt/`入面嘅文檔，開始下一個播放清單重複以上步驟。

## 上傳到 HF

1. 所有片都終於下載晒之後，`uv run 3_generate_metadata.py`，會生成一個 `metadata.csv`，作為 HF 上面數據集嘅超數據。
1. 按照 `HF_UPLOAD_STEPS.md` 入面步驟跑
    ```bash
    python 3_make_webdataset.py metadata.csv webdataset --dry-run
    ```
    會列出總共需要分成幾多個 tar 上傳到 HF 做數據集。確定冇問題之後刪去 `--dry-run` 再跑多次。
1. 


HF_XET_CACHE=/home/jupyter/xet-cache python 4_upload_to_hf.py --year 2025 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2024 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2023 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2022 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2021 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2020 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2019 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2018 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2017 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2016 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2015 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2014 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
python 4_upload_to_hf.py --year 2013 --repo_id laubonghaudoi/legco --config-name raw --staging-dir /home/jupyter/xet-cache/staging && \
