# Câu hỏi nghiên cứu 2: Các hàm loss giúp mô hình học tốt trên ảnh không bệnh như thế nào?

## 1. Mục tiêu nghiên cứu

Trong bài toán phân đoạn ảnh X-quang u xương, tập dữ liệu không chỉ gồm các ảnh có vùng tổn thương, mà còn có nhiều ảnh không bệnh với ground-truth mask rỗng. Với ảnh có bệnh, mô hình cần dự đoán đúng vị trí và hình dạng vùng tổn thương. Với ảnh không bệnh, mục tiêu lại khác: mô hình không nên sinh ra bất kỳ vùng dự đoán tổn thương giả nào.

Vì vậy, câu hỏi nghiên cứu đặt ra là:

> Các hàm loss khác nhau ảnh hưởng như thế nào đến khả năng giảm dự đoán dương tính giả trên ảnh không bệnh, trong khi vẫn giữ được chất lượng phân đoạn trên ảnh có bệnh?

Nghiên cứu này tập trung vào mô hình UNet ở kích thước ảnh 224×224, giữ cố định kiến trúc, dữ liệu, cách chia tập train/validation/test, augmentation, optimizer và scheduler. Yếu tố được thay đổi duy nhất là hàm loss. Cách thiết kế này giúp cô lập ảnh hưởng của loss function đối với hành vi của mô hình trên ảnh không bệnh.

---

## 2. Khảo sát lý thuyết các hàm loss

### 2.1. Đặc thù của ảnh không bệnh trong segmentation

Trong segmentation nhị phân, mỗi pixel được phân thành hai lớp:

- foreground: vùng tổn thương;
- background: vùng không tổn thương.

Với ảnh có bệnh, ground-truth mask chứa ít nhất một vùng foreground. Khi đó, các chỉ số như Dice hoặc IoU có ý nghĩa trực tiếp vì ta có thể đo mức độ chồng lấp giữa vùng dự đoán và vùng tổn thương thật.

Với ảnh không bệnh, ground-truth mask hoàn toàn rỗng. Toàn bộ pixel đều là background. Trường hợp này tạo ra một yêu cầu rất quan trọng: mô hình phải học rằng không nên dự đoán foreground ở bất kỳ vị trí nào. Nếu mô hình dự đoán một vùng nhỏ là tổn thương trên ảnh bình thường, đó là false positive trên ảnh không bệnh.

Trong bối cảnh y khoa, loại lỗi này đáng quan tâm vì mô hình có thể báo nhầm tổn thương cho bệnh nhân không có bệnh. Do đó, nếu chỉ đánh giá mô hình bằng Dice/IoU trên ảnh có bệnh thì chưa đủ. Cần có nhóm chỉ số riêng để đánh giá hành vi của mô hình trên ảnh không bệnh.

### 2.2. Dice Loss

Dice coefficient đo mức độ chồng lấp giữa mask dự đoán và mask thật:

```text
Dice = (2TP + eps) / (2TP + FP + FN + eps)
DiceLoss = 1 - Dice
```

Trong đó:

- TP: số pixel dự đoán đúng là foreground;
- FP: số pixel background bị dự đoán nhầm thành foreground;
- FN: số pixel foreground bị bỏ sót;
- eps: hằng số nhỏ để tránh chia cho 0.

Dice Loss rất phổ biến trong segmentation y khoa vì nó tối ưu trực tiếp độ chồng lấp vùng tổn thương. Loss này đặc biệt hữu ích khi foreground nhỏ hơn nhiều so với background, vì Dice không bị chi phối đơn thuần bởi số lượng pixel background lớn.

Tuy nhiên, Dice Loss có hạn chế khi xử lý ảnh không bệnh. Với ảnh không bệnh, ground-truth foreground rỗng. Khi đó bài toán không còn là “vùng dự đoán có chồng lấp với vùng bệnh thật không”, mà là “mô hình có dự đoán nhầm vùng bệnh nào không”. Dice Loss vốn được thiết kế quanh ý tưởng overlap foreground, nên tín hiệu học trên các ảnh mask rỗng có thể chưa đủ rõ để kiểm soát false positive.

Điều này dẫn đến giả thuyết: Dice-only có thể vẫn sinh nhiều mask giả trên ảnh không bệnh, dù có thể học được một phần vùng tổn thương trên ảnh có bệnh.

### 2.3. Binary Cross Entropy Loss

Binary Cross Entropy xem mỗi pixel là một bài toán phân loại nhị phân:

```text
BCE = -[y log(p) + (1-y) log(1-p)]
```

Trong đó:

- y là nhãn thật của pixel, bằng 0 hoặc 1;
- p là xác suất mô hình dự đoán pixel thuộc foreground.

Trong implementation thực nghiệm, BCE được dùng ở dạng `BCEWithLogits`, tức là loss nhận trực tiếp logits thay vì xác suất sau sigmoid. Cách này ổn định số học hơn so với tự sigmoid rồi mới tính BCE.

Đối với ảnh không bệnh, toàn bộ ground-truth mask bằng 0. Vì vậy, nếu mô hình dự đoán pixel nào có xác suất foreground cao, BCE sẽ phạt trực tiếp pixel đó. Đây là điểm quan trọng khiến BCE có lợi hơn Dice-only trong việc học trên ảnh không bệnh: nó tạo tín hiệu rõ ràng rằng tất cả pixel đều phải là background.

Tuy nhiên, nếu chỉ dùng BCE, mô hình có thể bị thiên về background vì số pixel background thường rất lớn so với foreground. Điều này có thể làm giảm khả năng phát hiện tổn thương nhỏ. Vì vậy, trong segmentation y khoa, BCE thường được kết hợp với Dice để cân bằng giữa giám sát ở mức pixel và tối ưu overlap vùng tổn thương.

### 2.4. BCE + Dice Loss

BCE + Dice kết hợp hai loại tín hiệu:

```text
Loss = w_bce * BCEWithLogits + w_dice * DiceLoss
```

Trong nghiên cứu này khảo sát hai cấu hình:

- BCE + Dice với trọng số 0.5/0.5;
- BCE + Dice với trọng số 0.7/0.3.

Cấu hình 0.5/0.5 được dùng làm baseline chính vì nó cân bằng giữa pixel-level supervision và region-level overlap. Cấu hình 0.7/0.3 tăng trọng số BCE, với kỳ vọng tạo tín hiệu mạnh hơn cho background pixels, từ đó giảm false positive trên ảnh không bệnh.

Về mặt trực giác:

- Dice giúp mô hình học vùng tổn thương trên ảnh có bệnh;
- BCE giúp mô hình học rằng ảnh không bệnh phải có mask rỗng;
- tăng trọng số BCE có thể giúp giảm mask giả, nhưng nếu quá mạnh có thể làm mô hình bảo thủ hơn và giảm recall.

Do đó, BCE + Dice là baseline hợp lý để so sánh với các loss chuyên biệt hơn.

### 2.5. Focal BCE + Dice Loss

Focal Loss là biến thể của BCE, được thiết kế để giảm ảnh hưởng của các pixel dễ và tập trung hơn vào các pixel khó:

```text
FocalBCE = - alpha_t * (1 - p_t)^gamma * log(p_t)
```

Trong đó:

- p_t là xác suất dự đoán đúng lớp thật;
- gamma điều khiển mức độ giảm trọng số của các pixel dễ;
- alpha_t điều chỉnh trọng số giữa positive và negative class.

Trong thực nghiệm này dùng:

```text
gamma = 2
alpha_positive = 0.25
```

Với quy ước `alpha_positive = 0.25`, pixel foreground có trọng số 0.25 và pixel background có trọng số tương ứng 0.75. Điều này phù hợp với mục tiêu giảm false positive, vì background/negative pixels được nhấn mạnh tương đối nhiều hơn.

Focal BCE + Dice có dạng:

```text
Loss = 0.5 * FocalBCE + 0.5 * DiceLoss
```

Kỳ vọng của cấu hình này là mô hình sẽ tập trung hơn vào các pixel background khó, tức là các vùng trên ảnh không bệnh hoặc vùng nền dễ bị nhầm thành tổn thương. Tuy nhiên, nếu nhấn mạnh negative/background quá nhiều, mô hình cũng có nguy cơ trở nên bảo thủ và bỏ sót tổn thương thật. Vì vậy cần đánh giá đồng thời normal false positive và tumor recall.

### 2.6. Tversky Loss

Tversky Loss là dạng tổng quát của Dice Loss, cho phép điều chỉnh riêng mức phạt cho false positive và false negative:

```text
Tversky = (TP + eps) / (TP + alpha_fp * FP + beta_fn * FN + eps)
TverskyLoss = 1 - Tversky
```

Trong đó:

- `alpha_fp` là trọng số phạt false positive;
- `beta_fn` là trọng số phạt false negative.

Nếu `alpha_fp` lớn hơn `beta_fn`, mô hình bị phạt mạnh hơn khi dự đoán nhầm background thành foreground. Đây là hướng phù hợp với mục tiêu giảm mask giả trên ảnh không bệnh.

Nghiên cứu khảo sát hai cấu hình:

```text
Tversky alpha_fp=0.7, beta_fn=0.3
Tversky alpha_fp=0.8, beta_fn=0.2
```

Hai cấu hình này đều phạt false positive mạnh hơn false negative. Tuy nhiên, cần thận trọng vì phạt false positive quá mạnh có thể khiến mô hình giảm dự đoán foreground nói chung. Khi đó normal false positive có thể giảm, nhưng tumor recall cũng có thể giảm. Vì vậy, Tversky cần được đánh giá bằng cả metric trên ảnh có bệnh và metric trên ảnh không bệnh.

### 2.7. Tổng hợp vai trò của các loss

| Loss | Tín hiệu chính | Kỳ vọng trên ảnh không bệnh | Rủi ro chính |
|---|---|---|---|
| Dice-only | Tối ưu overlap foreground | Có thể chưa kiểm soát tốt mask rỗng | Sinh false positive trên ảnh normal |
| BCE + Dice 0.5/0.5 | Cân bằng pixel-level và overlap | Baseline hợp lý, có tín hiệu background | Vẫn có thể còn false positive |
| BCE + Dice 0.7/0.3 | Tăng trọng số BCE/background | Có thể giảm false positive tốt hơn baseline | Có thể giảm recall nếu quá bảo thủ |
| FocalBCE + Dice | Tập trung vào pixel khó | Có thể xử lý vùng background dễ nhầm | Có thể bất ổn nếu class weighting chưa phù hợp |
| Tversky 0.7/0.3 | Phạt FP mạnh hơn FN | Có thể giảm dự đoán giả | Có thể bỏ sót tổn thương thật |
| Tversky 0.8/0.2 | Phạt FP rất mạnh | Có thể giảm FP mạnh hơn | Nguy cơ giảm recall cao hơn |

Từ khảo sát lý thuyết, một loss tốt cho bài toán này không chỉ cần tối ưu Dice trên ảnh có bệnh, mà còn cần kiểm soát false positive trên ảnh không bệnh. Vì vậy, đánh giá loss phải tách riêng hai nhóm ảnh: tumor cases và normal cases.

---

## 3. Thiết kế thực nghiệm ablation

### 3.1. Nguyên tắc thiết kế

Thực nghiệm được thiết kế như một ablation nhỏ, trong đó chỉ thay đổi hàm loss. Các thành phần còn lại được giữ cố định:

- mô hình: UNet Strong;
- kích thước ảnh: 224×224;
- dữ liệu và cách chia train/validation/test;
- augmentation;
- optimizer và scheduler;
- batch setup.

Việc giữ cố định các yếu tố trên giúp kết luận tập trung vào tác động của loss function.

Tập test không được dùng để chọn loss, checkpoint hoặc threshold. Toàn bộ lựa chọn trong giai đoạn ablation chỉ dựa trên validation set. Điều này giúp tránh test leakage và giữ test set cho đánh giá cuối cùng.

### 3.2. Các metric đánh giá

Metric được chia thành hai nhóm.

Nhóm 1: metric trên ảnh có bệnh, chỉ tính trên tumor cases:

- Tumor Dice;
- Tumor IoU;
- Precision;
- Recall.

Nhóm 2: metric trên ảnh không bệnh, chỉ tính trên normal cases:

```text
normal_fp_image_rate =
    số ảnh normal có predicted_area_ratio > 1e-4
    / tổng số ảnh normal
```

```text
normal_pred_area_ratio =
    trung bình predicted_area_ratio trên toàn bộ ảnh normal
```

Ngưỡng `1e-4` được dùng để tránh trường hợp chỉ vài pixel nhiễu rất nhỏ cũng bị xem là false positive nghiêm trọng. Với ảnh 224×224, ngưỡng này tương đương khoảng 5 pixel.

Hai metric normal quan trọng nhất trong câu hỏi nghiên cứu này là:

- `normal_fp_image_rate`: bao nhiêu ảnh không bệnh bị dự đoán nhầm là có vùng tổn thương;
- `normal_pred_area_ratio`: diện tích trung bình của vùng dự đoán nhầm trên ảnh không bệnh.

### 3.3. Quy tắc chọn threshold và checkpoint

Mỗi epoch được đánh giá trên validation set với nhiều threshold:

```text
0.30, 0.35, 0.40, ..., 0.70
```

Với mỗi epoch, trước tiên chọn threshold đại diện theo quy tắc normal-aware:

1. Chỉ xét các threshold có Tumor Dice không thấp hơn Dice tốt nhất của epoch quá 0.05.
2. Trong nhóm threshold này, ưu tiên:
   - normal_fp_image_rate thấp nhất;
   - nếu hòa, chọn normal_pred_area_ratio thấp nhất;
   - nếu vẫn hòa, chọn Tumor Dice cao nhất;
   - nếu vẫn hòa, chọn Recall cao nhất.

Sau khi mỗi epoch có threshold đại diện, các epoch được so sánh bằng cùng quy tắc trên. Checkpoint thắng được lưu kèm metadata:

- selected epoch;
- selected threshold;
- validation tumor metrics;
- validation normal metrics;
- selection rule.

Điểm quan trọng là threshold được chọn trên validation và sẽ được khóa lại. Test sau này bắt buộc dùng đúng checkpoint và threshold đã chọn, không sweep threshold lại trên test.

### 3.4. Các cấu hình loss trong screening E4

Giai đoạn E4 là screening chính, chạy 6 loss trong 30 epoch với seed 42:

| Ký hiệu | Loss | Vai trò |
|---|---|---|
| dice | Dice-only | Control để quan sát hạn chế của overlap-only loss |
| bce_dice_05 | 0.5 BCE + 0.5 Dice | Baseline chính |
| bce_dice_07 | 0.7 BCE + 0.3 Dice | Tăng tín hiệu background/BCE |
| focal_bce_dice | FocalBCE + Dice | Tập trung vào pixel khó |
| tversky_07_03 | Tversky alpha_fp=0.7, beta_fn=0.3 | Phạt FP mạnh hơn FN |
| tversky_08_02 | Tversky alpha_fp=0.8, beta_fn=0.2 | Phạt FP mạnh hơn nữa |

Baseline được xác định là `bce_dice_05`, và baseline này cũng được chọn checkpoint/threshold bằng cùng normal-aware validation rule. Challenger được xem là hợp lệ nếu so với baseline trên validation:

```text
Tumor Dice giảm không quá 0.05 tuyệt đối
Tumor Recall giảm không quá 0.10 tuyệt đối
```

Guard này nhằm tránh trường hợp một loss giảm false positive bằng cách học quá bảo thủ hoặc dự đoán gần như mask rỗng.

---

## 4. Kết quả thực nghiệm E4: Screening 6 loss

Giai đoạn E4 đã hoàn tất trên Kaggle GPU T4. Đây là kết quả screening validation của 6 loss:

| Loss | Selected epoch | Threshold | Tumor Dice | Recall | Precision | Normal FP rate | Normal area | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dice | - | - | - | - | - | 0.897 | 0.0696 | Control |
| bce_dice_05 | 14 | 0.65 | 0.2066 | 0.2089 | 0.5839 | 0.4113 | 0.00659 | Baseline |
| bce_dice_07 | 20 | 0.55 | 0.2632 | 0.2598 | 0.6973 | 0.2660 | 0.00179 | Có |
| focal_bce_dice | - | - | 0.2090 | 0.2471 | 0.4400 | 0.6135 | 0.00883 | Có |
| tversky_07_03 | - | - | 0.146 | - | - | 0.908 | - | Không |
| tversky_08_02 | - | - | 0.156 | - | - | 0.840 | - | Không |

Ghi chú: các dòng có dấu `-` là do báo cáo hiện tại chỉ trích các chỉ số chính đã được tổng hợp từ screening. Khi viết báo cáo cuối cùng, có thể thay bằng bảng CSV đầy đủ từ artifact E4.

### 4.1. So sánh baseline BCE+Dice 0.5/0.5 và BCE+Dice 0.7/0.3

Baseline `bce_dice_05` đạt:

```text
Tumor Dice = 0.2066
Tumor Recall = 0.2089
Normal FP rate = 0.4113
Normal area = 0.00659
```

Cấu hình `bce_dice_07` đạt:

```text
Tumor Dice = 0.2632
Tumor Recall = 0.2598
Normal FP rate = 0.2660
Normal area = 0.00179
```

So với baseline, `bce_dice_07` cải thiện đồng thời cả hai nhóm metric:

```text
Tumor Dice tăng: 0.2632 - 0.2066 = +0.0566
Tumor Recall tăng: 0.2598 - 0.2089 = +0.0509
Normal FP rate giảm: 0.4113 - 0.2660 = 0.1453
Normal area giảm: 0.00659 - 0.00179 = 0.00480
```

Tính tương đối, normal false positive image rate giảm khoảng:

```text
0.1453 / 0.4113 ≈ 35.3%
```

Normal predicted area ratio giảm khoảng:

```text
0.00480 / 0.00659 ≈ 72.9%
```

Kết quả này ủng hộ giả thuyết rằng tăng trọng số BCE trong BCE+Dice có thể giúp mô hình học tốt hơn trên ảnh không bệnh. Đáng chú ý, việc tăng BCE từ 0.5 lên 0.7 không làm giảm Dice/Recall trong screening này; ngược lại còn cải thiện cả Tumor Dice và Tumor Recall.

### 4.2. FocalBCE + Dice

`focal_bce_dice` vẫn được xem là hợp lệ theo guard Dice/Recall vì:

- Tumor Dice gần baseline;
- Tumor Recall không giảm quá ngưỡng cho phép.

Tuy nhiên, trên metric normal, cấu hình này kém hơn baseline:

```text
Normal FP rate = 0.6135
Normal area = 0.00883
```

Điều này cho thấy trong thiết lập hiện tại, FocalBCE + Dice chưa giúp giảm false positive trên ảnh không bệnh. Một khả năng là focal mechanism tập trung vào các pixel khó nhưng chưa trực tiếp tạo ra hành vi bảo thủ hơn trên normal cases. Ngoài ra, class weighting và gamma có thể cần tinh chỉnh thêm. Tuy vậy, vì mục tiêu của ablation là khảo sát loss trong phạm vi nhỏ, không mở rộng thêm hyperparameter search ở giai đoạn này.

### 4.3. Dice-only

Dice-only có normal false positive rất cao:

```text
Normal FP rate = 0.897
Normal area = 0.0696
```

Kết quả này phù hợp với phân tích lý thuyết. Dice-only tập trung vào overlap foreground và không cung cấp tín hiệu pixel-level đủ rõ cho ảnh mask rỗng. Khi dữ liệu có nhiều normal cases, Dice-only dễ sinh vùng dự đoán giả trên ảnh không bệnh. Vì vậy, Dice-only phù hợp làm control để minh họa hạn chế của overlap-only loss, nhưng không phù hợp làm baseline chính.

### 4.4. Tversky Loss

Hai cấu hình Tversky được kỳ vọng giảm false positive vì phạt FP mạnh hơn FN. Tuy nhiên, kết quả E4 cho thấy:

- `tversky_07_03` có Tumor Dice thấp hơn baseline quá ngưỡng guard;
- `tversky_08_02` cũng không đạt điều kiện hợp lệ;
- normal FP rate của hai cấu hình vẫn cao.

Điều này cho thấy trong thiết lập hiện tại, chỉ tăng trọng số phạt FP trong Tversky chưa đủ để cải thiện hành vi trên ảnh không bệnh. Một khả năng là Tversky vẫn là loss dựa trên region overlap, nên không tạo tín hiệu background pixel-level mạnh như BCE. Ngoài ra, phạt FP mạnh có thể làm quá trình học khó cân bằng hơn, đặc biệt khi tổn thương nhỏ và dữ liệu có độ biến thiên cao.

---

## 5. Kết luận từ E4

Kết quả screening E4 cho thấy BCE+Dice với trọng số 0.7/0.3 là cấu hình tốt nhất trong 6 loss được khảo sát. Cấu hình này không chỉ giảm false positive trên ảnh không bệnh mà còn cải thiện Tumor Dice và Tumor Recall so với baseline BCE+Dice 0.5/0.5.

Các phát hiện chính:

1. Dice-only không phù hợp để kiểm soát ảnh không bệnh vì normal false positive rất cao.
2. BCE+Dice là hướng phù hợp hơn Dice-only vì BCE cung cấp tín hiệu pixel-level cho background.
3. Tăng trọng số BCE từ 0.5 lên 0.7 giúp giảm rõ rệt false positive trên normal cases trong E4.
4. FocalBCE + Dice hợp lệ theo guard Dice/Recall nhưng không tốt bằng BCE+Dice 0.7/0.3 trên normal metrics.
5. Tversky với alpha_fp cao không đạt kỳ vọng trong thiết lập này và bị loại bởi guard Dice/Recall hoặc normal metrics.

Do đó, câu trả lời tạm thời cho câu hỏi nghiên cứu là:

> Trong thiết lập UNet 224×224, các loss có thành phần BCE rõ ràng giúp mô hình học tốt hơn trên ảnh không bệnh so với Dice-only hoặc Tversky-only. Trong các cấu hình đã khảo sát, BCE+Dice với trọng số 0.7/0.3 cho kết quả tốt nhất ở giai đoạn screening, vì vừa giảm normal false positive vừa không làm suy giảm khả năng phát hiện tổn thương.

---

## 6. Vai trò của E5 và E6 trong báo cáo

Do giới hạn thời gian tính toán trên Kaggle, E5 và E6 có thể được trình bày như giai đoạn kiểm định mở rộng.

### 6.1. E5: Full-stage nhiều seed

E5 được thiết kế để kiểm tra xem kết luận từ E4 có ổn định qua nhiều random seed hay không. Thay vì chỉ chạy seed 42, E5 chạy:

```text
BCE+Dice 0.5/0.5 × 5 seeds
BCE+Dice 0.7/0.3 × 5 seeds
```

Các seed:

```text
42, 52, 62, 72, 82
```

Mỗi seed có checkpoint và threshold riêng, được chọn bằng validation. Winner cuối cùng sẽ được chọn từ mean validation across seeds. Guard Dice/Recall vẫn được áp dụng để tránh chọn loss quá bảo thủ.

Nếu E5 chưa hoàn tất khi nộp báo cáo, có thể ghi:

> Giai đoạn screening E4 đã hoàn tất và cho thấy BCE+Dice 0.7/0.3 là challenger tốt nhất. Giai đoạn full-stage nhiều seed đang được thực hiện để kiểm tra độ ổn định của kết luận này. Do giới hạn thời gian tính toán, báo cáo hiện tại tập trung phân tích E4; kết quả E5/E6 sẽ được bổ sung trong phiên bản mở rộng.

### 6.2. E6: Locked test

E6 chỉ được chạy sau khi winner đã được khóa từ validation. Test set không được dùng để chọn loss, checkpoint hoặc threshold. Mục tiêu của E6 là đánh giá cuối cùng baseline và winner trên test set bằng checkpoint/threshold đã chọn từ validation.

E6 cũng sinh các prediction grids để minh họa định tính. Các hình này chỉ dùng để trình bày, không dùng để thay đổi lựa chọn loss.

---

## 7. Hạn chế hiện tại

Nghiên cứu hiện tại có một số hạn chế:

1. E4 mới là screening với một seed, nên kết luận cần được kiểm định thêm bằng E5 nhiều seed.
2. Chưa thực hiện hyperparameter search rộng cho FocalBCE hoặc Tversky; mỗi nhóm loss chỉ khảo sát một vài cấu hình đại diện.
3. Kết quả hiện tại chỉ áp dụng cho UNet 224×224 trong pipeline cố định; chưa kết luận cho các kiến trúc khác.
4. Test set chưa được dùng trong phần E4 vì test được giữ khóa cho đánh giá cuối cùng.

Tuy vậy, thiết kế validation-only giúp kết quả E4 vẫn có giá trị nghiên cứu: nó trả lời được loss nào là ứng viên tốt nhất để giảm false positive trên ảnh không bệnh trong phạm vi ablation hiện tại.

---

## 8. Kết luận chung

Ảnh không bệnh là một trường hợp quan trọng trong segmentation y khoa vì ground-truth mask rỗng, và lỗi cần tránh là mô hình sinh vùng tổn thương giả. Dice Loss, dù phổ biến trong segmentation, chưa đủ để kiểm soát tốt trường hợp này vì nó tập trung vào overlap foreground. BCE cung cấp tín hiệu pixel-level rõ ràng hơn cho background, do đó phù hợp hơn với ảnh không bệnh.

Thực nghiệm E4 trên 6 cấu hình loss cho thấy BCE+Dice 0.7/0.3 là lựa chọn tốt nhất trong screening. So với baseline BCE+Dice 0.5/0.5, cấu hình này tăng Tumor Dice và Tumor Recall, đồng thời giảm normal false positive image rate khoảng 35.3% và giảm normal predicted area ratio khoảng 72.9%. Kết quả này cho thấy việc tăng trọng số BCE là một hướng hiệu quả để giảm dự đoán dương tính giả trên ảnh không bệnh mà không làm suy giảm chất lượng phân đoạn trong giai đoạn screening.

Trong báo cáo hiện tại, E4 có thể được xem là kết quả thực nghiệm chính. E5 và E6 đóng vai trò kiểm định mở rộng: E5 kiểm tra độ ổn định qua nhiều seed, còn E6 đánh giá cuối cùng trên test set đã khóa.

