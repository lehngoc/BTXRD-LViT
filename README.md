# BTXRD-LViT

Repository cho khóa luận phân đoạn tổn thương xương từ X-quang theo hướng
patch-based, độ phân giải cao.

## Trạng thái hiện tại

Branch `phase0-protocol` là nền tảng protocol đã được freeze cho các giai
đoạn tiếp theo. Phase 0 quy định split dữ liệu chuẩn, manifests D_L/D_U,
contract không dùng GT trong inference, full-image metrics và locked test.

Các thí nghiệm 224x224 trước đây chỉ là prior work; chúng vẫn có thể tra cứu
qua Git history nhưng không phải workflow active của branch này.

## Nền tảng active

- Canonical split: `configs/splits/btxrd_split_seed42.csv`.
- Preprocessing/export: `configs/preprocess.yaml`, `scripts/01_prepare_btxrd.*`,
  `src/preprocessing/`, và `src/export/`.
- Phase 0 protocol: `configs/protocol/`, `src/protocol/`,
  `src/data/ssl_dataset.py`, và `src/evaluation/full_image_metrics.py`.
- Primitives tái sử dụng: `src/models/unet.py`, `src/training/losses.py`,
  `src/training/selection.py`, và `src/training/utils.py`.

## Phase 1 kế tiếp

Phase 1 là strong supervised Patch384 baseline trên D_L 50%. Cấu hình khởi
đầu được cố định ở patch 384, stride 192 và average aggregation. Model
selection chỉ dùng full-image validation; test chỉ chạy ở chế độ locked sau
khi checkpoint, threshold và config đã freeze. So sánh loss có kiểm soát là
BCE+Dice 0.5/0.5 và 0.7/0.3.

## Kiểm thử

Chạy toàn bộ unit tests từ thư mục repository:

```powershell
python -m pytest -q
```

## Cấu trúc chính

```text
configs/        Protocol, split và preprocessing configuration
docs/           Tài liệu dữ liệu và nghiên cứu
scripts/        Data-preparation entry points
src/            Data, protocol, evaluation, model và training primitives
tests/          Unit tests cho protocol và primitives
```
