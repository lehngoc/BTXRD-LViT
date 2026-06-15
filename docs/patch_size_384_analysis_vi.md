# Phân tích bounding box, lựa chọn Patch384 và thống kê dataset patch

## 1. Mục tiêu

BTXRD chứa ảnh X-quang có độ phân giải cao và kích thước không đồng nhất. Việc
resize toàn bộ ảnh xuống `224x224` giúp giảm chi phí tính toán nhưng có thể làm
mất chi tiết của vùng tổn thương nhỏ. Pipeline Patch384 giữ ảnh ở độ phân giải
đã tiền xử lý, lấy các crop cục bộ để huấn luyện và dùng sliding-window để suy
luận trên toàn ảnh.

Dataset được chia thành train/validation/test ở **cấp ảnh trước khi tạo patch**.
Sau đó, patch được sinh riêng từ từng manifest. Vì vậy các patch của cùng một
ảnh nguồn không thể xuất hiện ở nhiều split.

## 2. Phân tích bounding box tổn thương

### 2.1 Phương pháp

Phân tích sử dụng mask sau tiền xử lý trong
`data/exports/btxrd_preprocessed/all.csv`. Mỗi vùng foreground liên thông được
xem là một tổn thương:

- Connected components với 8-connectivity.
- Loại component có diện tích nhỏ hơn `10` pixel, đồng nhất với pipeline tạo
  patch.
- Bounding box được tính riêng cho từng component.
- Diện tích tổn thương là số pixel foreground thực tế của component, không phải
  diện tích hình chữ nhật bounding box.

Kết quả có `2,314` bounding box trên `1,867` ảnh u. Có `238` ảnh u chứa nhiều
bounding box, tương đương `12.75%`; số bounding box lớn nhất trên một ảnh là
`9`.

### 2.2 Thống kê kích thước

Đơn vị chiều dài là pixel trên ảnh sau tiền xử lý. Diện tích tổn thương có đơn
vị pixel vuông.

| Đại lượng | Min | Mean | Median | P75 | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Chiều rộng bbox | 10 | 182.98 | 118 | 227.75 | 435.70 | 1,230 |
| Chiều cao bbox | 11 | 246.74 | 188 | 322.75 | 532.70 | 1,448 |
| Cạnh lớn nhất bbox | 14 | 261.28 | 196 | 341.75 | 585.70 | 1,448 |
| Diện tích mask tổn thương | 119 | 50,305.38 | 14,666 | 48,372.25 | 146,929.80 | 1,303,390 |

Phân bố lệch phải rõ rệt: phần lớn tổn thương có kích thước vừa và nhỏ, trong
khi một nhóm nhỏ có kích thước rất lớn. Vì vậy chọn patch theo giá trị lớn nhất
sẽ làm tăng chi phí tính toán quá mức.

### 2.3 So sánh các patch size

`Chứa bbox với 20% context` nghĩa là cạnh lớn nhất của bbox không vượt quá 80%
kích thước patch, để còn không gian giải phẫu xung quanh tổn thương.

| Patch size | Chứa trọn bbox | Chứa bbox với 20% context | Median context margin | Pixel/patch so với 224 |
|---:|---:|---:|---:|---:|
| 224 | 56.70% | 46.20% | 14 px | 1.00x |
| 320 | 72.69% | 63.09% | 62 px | 2.04x |
| 384 | 78.91% | 71.18% | 94 px | 2.94x |
| 512 | 86.91% | 80.55% | 158 px | 5.22x |

Patch `384x384` được chọn vì:

- Lớn hơn P75 của cạnh lớn nhất (`341.75` pixel), nên chứa trọn phần lớn bbox.
- Tăng tỷ lệ chứa trọn bbox thêm `6.22` điểm phần trăm so với size 320.
- Cung cấp median context margin `94` pixel quanh bbox.
- Patch 512 chỉ tăng độ phủ thêm `8.00` điểm phần trăm so với 384 nhưng có số
  pixel mỗi patch lớn hơn khoảng `77.8%`, làm tăng bộ nhớ GPU cho mỗi sample.
- Các tổn thương lớn hơn 384 vẫn được quan sát qua nhiều crop grid và được ghép
  khi suy luận sliding-window.

Lựa chọn 384 là cân bằng giữa độ phủ tổn thương, ngữ cảnh giải phẫu và tài
nguyên huấn luyện. Đây chưa phải kết luận từ ablation Dice/IoU giữa nhiều patch
size.

## 3. Quy trình tạo Patch384

Pipeline được triển khai trong `src/export/extract_patches.py` và chạy riêng
cho ba manifest bằng `scripts/10_generate_patch384_dataset.*`. Các tham số hiện
tại:

```text
patch_size = 384
seed = 42
min_component_area = 10
jitter_fraction = 0.5
positive_crops_per_lesion = 3
hard_negatives_per_lesion = 1
random_negatives_per_image = 0.5
large_lesion_grid_stride_ratio = 0.75
max_large_lesion_grid_patches = 6
target_positive_ratio = 0.6
```

### 3.1 Chuẩn bị ảnh và tổn thương

Với mỗi dòng manifest, pipeline đọc ảnh màu và mask nhị phân sau tiền xử lý.
Ảnh u được tách thành các connected component 8-connectivity. Mỗi component có
bbox, tâm bbox, diện tích bbox, diện tích mask và cạnh lớn nhất.

Nếu chiều rộng hoặc chiều cao ảnh nhỏ hơn 384, ảnh được reflection padding và
mask được zero padding. Tọa độ crop được giới hạn trong biên ảnh đã pad, bảo đảm
mọi output đều có kích thước chính xác `384x384`.

### 3.2 Positive patch

Với mỗi component:

1. `positive_center`: một patch đặt tại tâm bbox.
2. `positive_jitter`: thêm hai patch có tâm dịch ngẫu nhiên:

```text
center_x += U(-0.5 * bbox_width,  +0.5 * bbox_width)
center_y += U(-0.5 * bbox_height, +0.5 * bbox_height)
```

3. `positive_large_lesion_grid`: nếu cạnh lớn nhất của bbox vượt 384, pipeline
   quét grid bên trong vùng tổn thương với stride `288` pixel (`0.75 * 384`) và
   giữ tối đa 6 patch/component.

Jitter làm đa dạng vị trí tổn thương trong crop. Grid giúp mô hình quan sát
nhiều phần của các tổn thương không thể nằm trọn trong một patch.

### 3.3 Negative patch và cân bằng lớp

- `hard_negative`: thử lấy một crop quanh mỗi bbox, với bán kính phụ thuộc kích
  thước bbox. Patch chỉ được giữ nếu mask crop hoàn toàn rỗng.
- `random_negative`: với tham số `0.5`, mỗi ảnh nhận 0 hoặc 1 lần thử lấy crop
  ngẫu nhiên, trung bình 0.5 lần/ảnh. Crop có mask không rỗng bị loại.
- `balance_negative`: sau khi sinh các patch trên, pipeline tiếp tục lấy crop
  rỗng ngẫu nhiên cho đến khi tỷ lệ positive gần `60%` hoặc đạt giới hạn số
  lần thử.

`patch_kind` mô tả cách patch được lấy mẫu. Nhãn huấn luyện `is_positive` được
tính lại từ mask thực tế trong crop. Do đó nhãn vẫn đúng nếu một crop mang tên
positive nhưng sau dịch chuyển hoặc giới hạn biên không còn chứa foreground.

### 3.4 Dữ liệu lưu cho mỗi patch

Mỗi patch tạo ra:

- Ảnh JPG và mask PNG nhị phân.
- `patch_id`, `image_id`, split và loại patch.
- Tọa độ crop, kích thước crop và đường dẫn ảnh/mask nguồn.
- Bbox nguồn, `bbox_coverage`, `mask_area`, `mask_coverage`.
- Thống kê cường độ ảnh và metadata văn bản phục vụ các thí nghiệm LViT.

### 3.5 Thông số sliding-window inference hiện tại

Implementation hiện hành nằm trong `src/inference/sliding_window.py`. Phép ghép
được thực hiện bằng cách cộng xác suất dự đoán của các cửa sổ rồi chia cho số
lần mỗi pixel được quan sát. Đây là **average probability merging**, không phải
Gaussian-weighted merging.

| Thông số | Giá trị hiện tại | Nguồn xác nhận |
|---|---|---|
| Window size | `384x384` | `configs/train_unet_patch384.yaml` |
| Stride | `192` pixel | Config và script đánh giá |
| Overlap danh nghĩa | `50%` | `1 - 192 / 384` |
| Padding | Không pad nếu ảnh lớn hơn window; ảnh nhỏ hơn window được reflect padding ở cạnh dưới/phải, hoặc edge padding cho kích thước suy biến | `pad_image_to_patch()` |
| Batch size inference | `4` | `sliding_window.batch_size` |
| Threshold mặc định | `0.5` | `metrics.threshold` |
| Merge method | Average probability | `prob_sum / count_sum` |
| Post-processing | Không có | Mask được tạo trực tiếp bằng `probability >= threshold` |

Window cuối trên mỗi trục được neo vào biên cuối của ảnh nếu stride không đi
đúng tới biên. Xác suất sau ghép được crop về kích thước ảnh gốc.

Notebook E2 có chạy threshold sweep trên validation với các giá trị
`0.3/0.4/0.5/0.6/0.7`, nhưng pipeline hiện không tự chọn threshold tối ưu.
Các lệnh đánh giá mặc định trong repo không truyền `--threshold`, nên sử dụng
`0.5`. Trường `sliding_window.merge` được truyền và kiểm tra khi inference,
nhưng implementation hiện chỉ hỗ trợ `average_probability`; cấu hình Gaussian
sẽ báo lỗi thay vì âm thầm dùng sai phương pháp ghép.

## 4. Thống kê dataset sau khi tạo patch

### 4.1 Thống kê theo split

`Ảnh nguồn trong patch` là số ảnh thực sự đóng góp ít nhất một patch. `Patch /
ảnh gốc` sử dụng tổng số ảnh trong manifest làm mẫu số.

| Split | Ảnh nguồn trong patch / manifest | Patch | Positive | Negative | Positive ratio | Patch / ảnh gốc |
|---|---:|---:|---:|---:|---:|---:|
| Train | 2,393 / 2,622 | 8,862 | 5,317 | 3,545 | 60.00% | 3.380x |
| Validation | 519 / 562 | 2,041 | 1,224 | 817 | 59.97% | 3.632x |
| Test | 520 / 562 | 1,927 | 1,156 | 771 | 59.99% | 3.429x |
| Tổng | 3,432 / 3,746 | 12,830 | 7,697 | 5,133 | 59.99% | 3.425x |

Tất cả `1,867` ảnh u đều xuất hiện trong patch dataset. Ảnh thường có độ phủ
thấp hơn, khoảng `82.6%` đến `85.1%`, vì random negative được lấy theo xác suất
và balance negative không bắt buộc chọn mọi ảnh thường.

| Split | Ảnh u | Ảnh thường | Ảnh u có patch | Ảnh thường có patch | Coverage ảnh thường |
|---|---:|---:|---:|---:|---:|
| Train | 1,307 | 1,315 | 1,307 | 1,086 | 82.59% |
| Validation | 280 | 282 | 280 | 239 | 84.75% |
| Test | 280 | 282 | 280 | 240 | 85.11% |
| Tổng | 1,867 | 1,879 | 1,867 | 1,565 | 83.29% |

### 4.2 Phân bố loại patch

| Loại patch | Số lượng | Tỷ lệ |
|---|---:|---:|
| `positive_jitter` | 4,628 | 36.07% |
| `balance_negative` | 3,401 | 26.51% |
| `positive_center` | 2,314 | 18.04% |
| `random_negative` | 1,540 | 12.00% |
| `positive_large_lesion_grid` | 757 | 5.90% |
| `hard_negative` | 190 | 1.48% |

Có hai patch mang loại lấy mẫu positive nhưng mask crop thực tế rỗng: một
`positive_jitter` ở train và một `positive_large_lesion_grid` ở test. Hai patch
này được gán `is_positive = 0`, nên nhãn sử dụng khi huấn luyện vẫn nhất quán
với mask.

Kiểm tra trực quan xác nhận đây không phải lỗi lệch tọa độ. Cả hai crop đều
giao với hình chữ nhật bbox nhưng rơi vào vùng rỗng của component có hình dạng
không lấp đầy bbox:

- Train jitter:
  `data/processed/reports/patch384_dataset_analysis/positive_named_empty_previews/IMG000402__lesion03__positive_jitter_02.jpg`
- Test large-lesion grid:
  `data/processed/reports/patch384_dataset_analysis/positive_named_empty_previews/IMG001528__lesion01__positive_large_lesion_grid_04.jpg`

Trong dataset hiện tại, hai crop được giữ như negative nhờ `is_positive = 0`.
Pipeline tạo patch đã được chỉnh để các lần sinh sau ghi loại này là
`empty_positive_proposal`, đồng thời lưu loại đề xuất ban đầu trong
`proposed_patch_kind`, tránh gọi chúng là positive trong thống kê phương pháp
lấy mẫu.

### 4.3 Mất cân bằng pixel trong positive patch

Mặc dù positive patch chiếm gần `60%` số patch, foreground vẫn chiếm tỷ lệ nhỏ
trong nhiều positive patch.

| Đại lượng trên 7,697 positive patch | Giá trị |
|---|---:|
| Mean `mask_area` | 35,358.56 pixel |
| Median `mask_area` | 18,766 pixel |
| P25 / P75 / P90 `mask_area` | 6,007 / 51,551 / 100,745.60 pixel |
| Mean tỷ lệ foreground | 23.98% |
| Median tỷ lệ foreground | 12.73% |
| P10 / P25 / P75 / P90 tỷ lệ foreground | 1.41% / 4.07% / 34.96% / 68.32% |
| Foreground dưới 1% diện tích patch | 527 patch, 6.85% |
| Foreground dưới 5% diện tích patch | 2,209 patch, 28.70% |
| Foreground chạm ít nhất một biên patch | 3,352 patch, 43.55% |

Các số liệu này cho thấy cân bằng positive/negative ở cấp patch không loại bỏ
mất cân bằng ở cấp pixel. Gần một phần ba positive patch vẫn có foreground dưới
5%, và hơn 40% có mask chạm biên do jitter hoặc chia tổn thương lớn.

### 4.4 Chất lượng coverage theo loại positive patch

Các hàng dưới đây chỉ tính trên patch có `is_positive = 1`. `Mask coverage` là
tỷ lệ foreground trên diện tích patch. `BBox coverage < 50%` là tỷ lệ patch chỉ
giao với dưới một nửa diện tích bbox nguồn.

| Loại patch | Count | Mean mask coverage | Median | P10 | Mean bbox coverage | BBox coverage < 50% |
|---|---:|---:|---:|---:|---:|---:|
| `positive_center` | 2,314 | 23.25% | 10.75% | 1.30% | 91.89% | 7.74% |
| `positive_jitter` | 4,627 | 17.68% | 10.31% | 1.30% | 85.75% | 13.10% |
| `positive_large_lesion_grid` | 756 | 64.75% | 66.98% | 31.19% | 50.69% | 55.82% |

Center patch có độ phủ bbox cao nhất. Jitter giảm coverage để tăng đa dạng vị
trí và ngữ cảnh. Large-lesion grid có bbox coverage thấp theo thiết kế vì mỗi
patch chỉ quan sát một phần bbox lớn hơn 384; đồng thời nhóm này có mask
coverage cao nhất.

### 4.5 Kiểm tra chất lượng và sử dụng khi huấn luyện

Utility tổng hợp kiểm tra:

- Tổng số dòng metadata khớp `report.json` của từng split.
- Mỗi dòng metadata có đủ file ảnh và mask.
- Không có `image_id` giao nhau giữa train, validation và test.
- Phân bố `patch_kind`, nhãn thực tế và mask coverage.
- Foreground chạm biên, positive patch có foreground rất nhỏ và coverage theo
  loại patch.
- Xuất preview chẩn đoán cho các positive proposal có mask rỗng.

`BTXRDPatchSegmentationDataset` kiểm tra kích thước patch, chuyển ảnh sang RGB,
chuẩn hóa ImageNet và chuyển mask thành tensor nhị phân. Smoke test kiểm tra
shape ảnh `(3, 384, 384)`, shape mask `(1, 384, 384)`, forward pass và backward
pass của UNet.

## 5. Phiên bản U-Net Patch384 và các cấu hình chính

### 5.1 Định danh thí nghiệm

Phiên bản chính được cấu hình trong `configs/train_unet_patch384.yaml`:

| Nhóm | Cấu hình | Giá trị |
|---|---|---|
| Experiment | Tên | `E2_unet_patch384_preprocessed_pos060` |
| Input | Kích thước patch | `384x384`, RGB |
| Dataset | Train / validation / test patch | `8,862 / 2,041 / 1,927` |
| Dataset | Tỷ lệ positive patch mục tiêu | `60%` |
| Model | Kiến trúc | U-Net 2D, 4 mức encoder/decoder |
| Model | Số kênh | `32 -> 64 -> 128 -> 256 -> 512` |
| Model | Số tham số trainable | `7,763,041` |
| Output | Phân đoạn | Một kênh logit nhị phân, cùng kích thước input |
| Chuẩn hóa | Ảnh input | ImageNet mean/std |
| Seed | Tái lập | `42` |

Tên `pos060` mô tả tỷ lệ positive patch khoảng `0.60`, không phải tỷ lệ pixel
foreground. Trên train patch, foreground chỉ chiếm trung bình `14.35%` tổng số
pixel.

Metadata patch có trường `text_lvit_prompt`, nhưng pipeline E2 khởi tạo dataset
với `include_text=False` vì U-Net chỉ nhận tensor ảnh RGB. Vì vậy text metadata
**không được nạp và không tham gia** kết quả của phiên bản E2 này.

### 5.2 Kiến trúc mô hình

U-Net sử dụng bốn lần max-pooling, bottleneck và bốn khối upsampling có skip
connection:

- Mỗi `ConvBlock` gồm hai lớp convolution `3x3`, mỗi lớp theo sau bởi
  BatchNorm và ReLU.
- Downsampling dùng MaxPool `2x2`; upsampling dùng transposed convolution
  `2x2`.
- Với input `384x384`, feature map tại bottleneck có kích thước `24x24` và
  `512` kênh.
- Head là convolution `1x1`, sinh một logit cho mỗi pixel. Sigmoid chỉ được áp
  dụng trong loss hoặc lúc đánh giá.

Docstring trong `src/models/unet.py` gọi đây là baseline cho `224x224`, nhưng
implementation là fully convolutional và forward pass đã được smoke test với
input/output lần lượt `(1, 3, 384, 384)` và `(1, 1, 384, 384)`.

### 5.3 Cấu hình huấn luyện

| Cấu hình | Giá trị | Ý nghĩa |
|---|---:|---|
| Batch size | `4` | Khoảng `2,216` step/train epoch |
| Epoch tối đa | `100` | Giới hạn trên của quá trình train |
| Optimizer | AdamW | Không cấu hình learning-rate scheduler |
| Learning rate | `3e-4` | Cố định trong suốt quá trình train |
| Weight decay | `1e-4` | Regularization qua AdamW |
| Loss | BCEWithLogits + Dice | Hai thành phần có trọng số `1:1` |
| BCE `pos_weight` | `20.0` | Tăng trọng số **pixel foreground** |
| Dice scope | Positive patch | Không tính DiceLoss trên negative patch |
| Patch validation | Mỗi epoch | Dùng để theo dõi nhanh, không chọn best model |
| Full-image validation | Epoch `1`, mỗi `5` epoch và epoch cuối | Sliding-window trên ảnh validation đầy đủ |
| Chọn checkpoint | Full-image validation tumor Dice cao nhất | Lưu thành `best.pt` |
| Full-validation patience | `20` lần full-image validation không cải thiện | Không phải 20 epoch |

Loss thực tế là:

```text
loss = BCEWithLogits(pos_weight=20) + DiceLoss(positive patches only)
```

Negative patch vẫn tham gia BCE để kiểm soát false positive, nhưng không tham
gia DiceLoss. Trên train set, tỷ lệ background/foreground pixel xấp xỉ `5.97`.
Với `pos_weight=20`, tổng trọng số lý thuyết của foreground trong BCE cao hơn
background khoảng `3.35x`. Cấu hình này ưu tiên recall mạnh và có thể làm tăng
false positive; cần đánh giá bằng `normal_fp_image_rate` và ablation
`pos_weight`, không nên kết luận chỉ từ Dice trên positive patch.

Pipeline hiện không có augmentation online, learning-rate scheduler, mixed
precision, gradient clipping hoặc post-processing. Jitter chỉ xuất hiện khi
tạo sẵn dataset patch, nên mỗi epoch đọc lại cùng tập crop.

`full_val_patience=20` chỉ tăng sau một lần full-image validation không
cải thiện. Với `full_val_interval=5` và tối đa `100` epoch, cấu hình hiện tại có
21 lần full-image validation; do đó early stopping gần như không rút ngắn quá
trình train trước epoch cuối.

## 6. Giao thức báo cáo kết quả mô hình

### 6.1 Chỉ số chính cần báo cáo

Kết quả chính phải lấy từ full-image sliding-window, vì đây là điều kiện sử
dụng thực tế của mô hình patch:

| Nhóm ảnh | Chỉ số nên báo cáo |
|---|---|
| Ảnh u | `tumor_dice`, `tumor_iou`, `tumor_precision`, `tumor_recall` |
| Ảnh thường | `normal_fp_image_rate`, `normal_pred_area_ratio` |
| Hiệu năng | `avg_windows_per_image`, `seconds_per_image` |
| Cấu hình inference | threshold, patch size, stride, overlap, merge, post-processing |

Các metric được tính riêng trên từng ảnh rồi lấy trung bình, không phải cộng
toàn bộ TP/FP/FN của dataset trước khi tính. `normal_fp_image_rate` xem một ảnh
thường là false positive nếu vùng dự đoán chiếm ít nhất `0.1%` diện tích ảnh.

Không nên dùng `all_dice` làm kết quả chính vì ảnh thường có mask rỗng có thể
làm chỉ số tổng hợp trông tốt hơn. Patch-level validation/test chỉ nên dùng để
chẩn đoán quá trình train; checkpoint được chọn bằng
`full-image sliding-window validation tumor_dice`.

### 6.2 Quy trình báo cáo đề xuất

1. Train tối đa 100 epoch và chọn `best.pt` duy nhất bằng tumor Dice trên
   full-image validation.
2. Nếu tối ưu threshold, chỉ sweep trên validation; sau đó khóa threshold trước
   khi chạy test.
3. Báo cáo validation và test với cùng patch size, stride, merge,
   post-processing và threshold.
4. Báo cáo đồng thời chất lượng phân đoạn ảnh u và false positive trên ảnh
   thường.
5. Ghi rõ đây là kết quả của một seed (`42`) nếu chưa chạy nhiều seed.

### 6.3 Trạng thái kết quả hiện tại

Hiện chưa có thư mục hoặc artifact
`experiments/E2_unet_patch384_preprocessed_pos060` trong workspace. Vì vậy chưa
thể báo cáo Dice/IoU/precision/recall hay tốc độ inference thực nghiệm cho E2.
Các số liệu ở trên là phân tích cấu hình, dataset và hành vi implementation,
không phải kết quả chất lượng mô hình.

Sau khi chạy đầy đủ, bộ artifact tối thiểu để lập bảng kết quả gồm:

- `config.json`, `history.csv`, `best_summary.json`, `best.pt` và `last.pt`.
- Các file `val_sliding_epoch_XXX.json` dùng để theo dõi quá trình chọn model.
- `test_patch_metrics.json` chỉ dùng cho chẩn đoán patch-level.
- `val_sliding_metrics.json` và `test_sliding_metrics.json` dùng cho báo cáo
  chính.

## 7. Tái tạo kết quả

Sinh lại dataset patch:

```bash
scripts/10_generate_patch384_dataset.sh
```

Phân tích bounding box và so sánh patch size:

```bash
python src/analysis/analyze_lesion_bbox.py --min-component-area 10
```

Tổng hợp và kiểm tra dataset Patch384:

```bash
python src/analysis/summarize_patch_dataset.py
```

Chạy smoke test:

```bash
python src/training/smoke_test_patch_pipeline.py --config configs/train_unet_patch384.yaml
```

Các báo cáo sinh tự động được lưu cục bộ trong:

```text
data/processed/reports/lesion_bbox_analysis/
data/processed/reports/patch384_dataset_analysis/
```
