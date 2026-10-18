# Báo cáo rà soát artifact đặc trưng

## NSL-KDD

- Số đặc trưng: 39, F1 (full features) = 0.7532
- Top 5 importance:
    - `flag`: 0.5791
    - `protocol_type`: 0.1430
    - `dst_host_same_srv_rate`: 0.0819
    - `service`: 0.0267
    - `wrong_fragment`: 0.0244
- ⚠ Đặc trưng bị cảnh báo và kết quả ablation:
    - `flag`: F1 có=0.7532, không có=0.7267, chênh lệch=+0.0265

## CICIDS2017

- Số đặc trưng: 52, F1 (full features) = 0.9943
- Top 5 importance:
    - `Bwd Packet Length Std`: 0.5088
    - `Max Packet Length`: 0.1094
    - `Average Packet Size`: 0.1061
    - `Destination Port`: 0.0765
    - `Init_Win_bytes_backward`: 0.0469
- ⚠ Đặc trưng bị cảnh báo và kết quả ablation:
    - `Bwd Packet Length Std`: F1 có=0.9943, không có=0.9943, chênh lệch=+0.0000

## UNSW

- Số đặc trưng: 42, F1 (full features) = 0.8631
- Top 5 importance:
    - `sttl`: 0.6679
    - `ct_srv_dst`: 0.0640
    - `sbytes`: 0.0525
    - `smean`: 0.0313
    - `ct_srv_src`: 0.0158
- ⚠ Đặc trưng bị cảnh báo và kết quả ablation:
    - `sttl`: F1 có=0.8631, không có=0.8656, chênh lệch=-0.0025
