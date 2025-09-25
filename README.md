# GigaCan

VAD 模型用 silero-vad，語音識別用 [Qwen3 ASR](https://qwen.ai/blog?id=41e4c0f6175f9b004a03a07e42343eaaf48329e7&from=research.latest-advancements-list)

- [api key](https://bailian.console.aliyun.com/?tab=model#/api-key)
- 模型 [qwen3-asr-flash](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/group-qwen3-asr-flash?modelGroup=group-qwen3-asr-flash)
- [阿里雲餘額](https://billing-cost.console.aliyun.com/fortune/billing-account)

## 流程

### 1 下載影片

呢一步要反覆試錯，因為 yt-dlp 下載經常會中斷。而且需要好多儲存空間，所以要經常𥄫住然後人手重試。首先要確定目標頻道或者播放清單，然後針對

1. 準備好 GCP 嘅 YouTube API key，然後跑`1_get_video_list.py` 將指定頻道或者播放清單入面所有影片嘅 metadata 爬落一個 csv 文件度，呢個文件亦都會用於登記下載進度。
1. 跑 `2_download_audio.py`，會按照上面嘅 csv 記錄嘅進度，將未下載嘅片下載落`download/`並轉化成 16kHz OPUS 格式。
    1. 因為下載過程會經常因為 YouTube 反爬蟲、空間唔夠等等意外中斷，所以需要有呢個 csv 嚟記錄進度。如果下載中斷，可以跑 `scan_progress.py`，會自動檢查 `download/` 入面邊啲已經下載咗邊啲未下載，然後更新個 csv 將啲已經下載且轉碼成功嘅登記為 `downloaded=True`。
    1. 每次中斷後重新跑 `2_download_audio.py` 都會自動讀取個 csv，按照 `downloaded`嗰列 `false` 嘅嚟下載。
    1. 下載完之後跑個 `2_organize_downloads.py` 會自動將 `download/` 入面下載好嘅音頻按照年份分類。
1. 全部下載完成之後，跑一次
    ```bash
    python3 2_check_audio_integrity.py download/ legco_20250920.csv --cleanup --auto-yes
    ```
    會自動按照個 csv 檢查一次所有下載好嘅 OPUS。如果遇到個長度唔對應嘅，就會刪除呢條 OPUS 然後喺 csv 入面標記`downloaded=False`
1. 再重複跑 `2_download_audio.py`，直至將所有 opus 都下載齊為止。
1. 跑 `3_generate_metadata.py`，會生成一個 `metadata.csv`，作為 HF 上面數據集嘅超數據。
1. 按照 `HF_UPLOAD_STEPS.md` 入面步驟跑
    ```bash
    python 3_make_webdataset.py metadata.csv webdataset --dry-run
    ```
    會列出總共需要分成幾多個 tar 上傳到 HF 做數據集。
1. 

## 2 轉寫字幕

1. 針對影片類別修改 system  prompt，然後跑 `2_vtt.py`，會用 silero-vad 將輸入音頻分段再叫 qwen3-asr-flash 轉寫成粵文，生成 .vtt 字幕文件到 `vtt/`。
    1. 記得修改 `2_vtt.py` 入面嘅 prompt，會對字幕準確度有好大影響。
    1. 唔同題材需要設定唔同嘅`--vad-merge-ms`時長，例如張悦楷三國演義最優大概係 450，而毛澤東的黃昏歲月就最好係 500。推薦每加一個新題材之前用`tune_vad.ipynb`嚟確定最優值。
1. 跑 `3_cut.py` 用生成嘅字幕分割音頻，生成切割後音頻到`cut/`入面，同時生成對應 `metadata.csv`
1. 讀取 `cut/` 入面切分好嘅音頻，上傳數據集到 HuggingFace
1. 刪除 `download`/ 同埋 `vtt/`入面嘅文檔，開始下一個播放清單重複以上步驟。

下面係命令。跑之前要創建 venv 安裝好依賴，然後開個 `.env` 放阿里雲個 `API_KEY=` 入去。跟住你要去 GCP 開個 YouTube API key，放個 `YOUTUBE_API_KEY` 嚟獲取 YouTube 影片數據。

```bash

# 下載播放列表或者頻道所有視頻
python3 1_get_video_list.py
# 修改 system prompt 並且調整 --vad-merge-ms，然後生成字幕
python3 2_vtt.py
# 用生成嘅 vtt 字幕切分
python3
```
