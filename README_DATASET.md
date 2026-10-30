# Quy trình lọc Grasp-Anything++ từ split và dựng lại dataset

Tài liệu này mô tả **các bước đã dùng trước đây**, phân biệt bản manifest thử nghiệm
với baseline hiện tại, và hướng dẫn dựng lại từ ZIP bằng
[`tools/rebuild_grasp_anything_pp.py`](tools/rebuild_grasp_anything_pp.py).

Script mới không đọc dữ liệu từ `grasp-anything-pp-selected`, không dùng manifest cũ
để quyết định mẫu, không sửa ZIP/split/dataset đang có. Mặc định ghi vào thư mục mới
`data/grasp-anything-pp-rebuilt`. Chỉ chạy khi bạn tự gọi lệnh; việc tạo script không
đồng nghĩa đã chạy lại dữ liệu.

## 1. Phân biệt scene, object, sample và grasp

| Cấp | Ví dụ tên | Ý nghĩa |
|---|---|---|
| Scene | `<scene_id>.jpg` | Một ảnh; `scene_id` là chuỗi hash 64 ký tự |
| Object ID | `<scene_id>_<object_id>` | Một đối tượng trong ảnh; đây là ID lưu trong các file split `.obj` |
| Part sample | `<scene_id>_<object_id>_<part_id>` | Một mẫu positive grasp/instruction/part mask |
| Grasp rectangle | Một hàng trong tensor `.pt` | Một phương án grasp; một part sample có thể chứa nhiều rectangle |

Do đó, **số object ID không bằng số ảnh, số part sample hay số rectangle**.
Một ảnh có thể xuất hiện trong nhiều mẫu object/part.

## 2. Các split chính thức được dùng

Đối với `--dataset grasp-anywhere`, nguồn ID là Grasp-Anything++:

| Nhóm | File split | Object ID | Ảnh phân biệt trong danh sách ID |
|---|---|---:|---:|
| Train seen pool | `split/grasp-anything++/train/seen.obj` | 14.516 | 11.616 |
| Test seen | `split/grasp-anything++/test/seen.obj` | 573 | 420 |
| Test unseen | `split/grasp-anything++/test/unseen.obj` | 230 | 200 |

Các `.obj` này là dữ liệu pickle chứa ID, không phải mesh 3D. Script đọc danh sách
plain IDs, kiểm tra định dạng và không cho phép pickle khởi tạo Python class.
**Không dùng `split/grasp-anything/seen.obj` để thay cho split ++**.

## 3. Ghép những file nào từ ZIP nào?

Với sample `<scene>_<obj>_<part>`:

| Nguồn | File cần lấy | Khóa ghép |
|---|---|---|
| ++ `grasp_label_positive.zip` | `grasp_label_positive/<scene>_<obj>_<part>.pt` | Object prefix phải thuộc split |
| ++ `grasp_instructions.zip` | `grasp_instructions/<scene>_<obj>_<part>.pkl` | Đúng sample có positive label |
| ++ `part_mask.zip` | `part_mask/<scene>_<obj>_<part>.npy` | Đúng sample có positive label |
| Grasp-Anything gốc `scene_description.zip` | `scene_description/<scene>.pkl` | Scene được positive samples tham chiếu |
| Grasp-Anything gốc `mask.zip` | `mask/<scene>_<obj>.npy` | Object có positive label; file này tùy chọn |
| Grasp-Anything gốc `image.zip` hoặc ảnh đã giải nén | `image/<scene>.jpg` | Scene được positive samples tham chiếu |

`grasp_label_negative.zip` chỉ được lấy nếu thêm `--include-negative`; nhãn negative
không thay cho positive và không được tính thành mẫu train/test positive.
Không lấy `grasp_label_positive.zip` của Grasp-Anything gốc thay cho ZIP ++: chúng
có tên ZIP giống nhau nhưng đơn vị annotation khác nhau.

Script lấy **positive label làm mốc trước**, rồi ghép companion theo **đúng sample ID**.
Không lấy tất cả part của object nếu part đó không có positive label. Phần tách ID
của script hỗ trợ object/part nhiều chữ số, không đoán bằng cách cắt một chữ số.

## 4. Các bước đã làm trước đây

### 4.1. Lọc theo membership của split và lấy file tương ứng

Trước đây, ảnh theo ID được chọn từ bộ ảnh gốc. Positive label, instruction và
part mask được lấy từ ZIP ++; object mask và scene description lấy từ bộ gốc.
Dữ liệu tách theo `train_seen`, `test_seen`, `test_unseen`.

Số liệu của bản dữ liệu đã chuẩn bị trước lần dựng lại mới:

| Nhóm | Object ID có positive label | Object ID chưa có positive label | Part samples có positive |
|---|---:|---:|---:|
| Train seen pool | 10.130 / 14.516 | 4.386 | 28.621 |
| Test seen | 515 / 573 | 58 | 1.376 |
| Test unseen | 212 / 230 | 18 | 601 |

Tổng là **30.598 part samples**. Đây là **số liệu đã ghi nhận của bộ cũ**, không phải
số lượng script mới ép phải đạt. Kết quả dựng lại phải lấy từ `report.json` mới.
Các số liệu này không chứng minh ID thiếu vẫn có trong ZIP: cần quét ZIP để xác nhận.

Bản folder dùng được khi đó có 412 ảnh test seen và 193 ảnh test unseen, so với
420/200 scene được split nhắc tới. Một scene không có positive annotation được
chọn có thể không được đưa vào folder model. Vì vậy, ảnh không có trong folder
đã chuẩn bị **không tự động có nghĩa** ảnh đó không tồn tại trong bộ ảnh gốc.

### 4.2. Nhánh manifest thử nghiệm đã có thêm bước lọc khác

`tools/prepare_grasp_anything_pp.py` đã từng tạo
`data/grasp-anything-pp-ready/manifest.jsonl`. Nhánh đó kiểm tra nội dung ảnh/mask,
đơn vị grasp, chất lượng rasterize và chia validation theo scene.
Báo cáo cũ ghi nhận:

- Loại 362 mẫu do chưa xác minh được đơn vị grasp subpixel.
- Loại 89 mẫu train thuộc 37 scene dùng chung với test.
- Kết quả: train 25.404, val 2.783, test seen 1.363, test unseen 597 mẫu.

**Đây không phải pool/split của baseline hiện tại.** Script dựng lại trong tài liệu
này không áp dụng lại các bước loại mẫu hoặc scene đó. Nó không tạo manifest cho
`grasp-anything-language`, không sửa tọa độ/đơn vị grasp, không bỏ grasp row.

### 4.3. Pool baseline hiện tại

Sau đó, các positive samples đã lấy theo split được ghép thành layout của loader
gốc: `grasp-anything` chứa ảnh/scene description, `grasp-anything-pp` chứa annotation.
Các hardlink được dùng để tránh sao chép thêm toàn bộ dữ liệu.

Loader chọn ID trong `train/seen.obj`, sắp xếp danh sách positive files, rồi trainer
chia theo **sample index**, với mặc định:

```text
N = 28.621
cut = floor(0,9 × N) = 25.758
train = danh sách index trước cut
val   = danh sách index từ cut trở đi = 2.863
```

Đây không phải chia 90/10 theo object, ảnh hay scene. Nếu `--ds-shuffle` bật, NumPy
shuffle danh sách index trước khi cắt. Nếu không bật, membership train/val lấy từ
thứ tự file đã sort. `SubsetRandomSampler` vẫn xáo thứ tự lấy batch.

`--ds-rotate` dịch danh sách file, **không phải góc xoay ảnh**. Xoay/zoom ảnh là
augmentation lúc load batch. Dataset dùng chung cho train và validation trong
trainer baseline đang giữ `random_rotate=True`, `random_zoom=True` như code gốc;
script lọc dữ liệu không tạo/lưu ảnh augment. Các dataset test được tạo riêng,
tắt rotate/zoom trong evaluator.

## 5. Script dựng lại mới làm gì?

1. Đọc lại cả ba file split chính thức; lưu SHA-256 của split.
2. Quét `grasp_label_positive.zip` ++, lấy mọi positive sample có object ID trong
   split tương ứng vào `selected/<nhóm>/`.
3. Quét instruction/part-mask ZIP và chỉ lấy đúng những sample đã tìm thấy positive.
4. Lấy ảnh, scene description và object mask theo scene/object thực sự được
   positive labels tham chiếu. Không cần copy ảnh của object không có positive label.
5. Khi giải nén, kiểm tra size/CRC, ghi file tạm rồi đổi tên. File đầu ra có sẵn
   được kiểm tra trước; file khác nội dung không bị ghi đè.
6. Xuất CSV provenance: archive, member name, CRC, kích thước và output path;
   xuất inventory annotation và danh sách object thiếu positive/ảnh bắt buộc.
7. Sinh các danh sách sample index mô phỏng đúng cách chia của baseline.
8. Chỉ xuất layout model sau khi tất cả positive samples đã chọn có ảnh,
   instruction, part mask và scene description. Object mask không bắt buộc.
   Nếu thiếu companion, ghi báo cáo rồi dừng, không âm thầm bỏ positive sample.
9. Layout kết hợp dùng hardlink mặc định hoặc copy theo `--link-mode copy`.

Object ID không có positive trong ZIP vẫn được ghi rõ trong báo cáo và danh sách
thiếu; script không sinh nhãn thay thế. Các scene dùng chung giữa split cũng được
báo cáo, **không tự loại để thay đổi protocol baseline**.

### Giới hạn tên file của loader baseline

Loader hiện giữ cách lọc gốc `filename[:-5]` và regex RGB chỉ hỗ trợ một chữ số cho
object/part. Bộ lọc ZIP mới vẫn giữ đúng dữ liệu gốc, kể cả ID nhiều chữ số.

- `*_raw_positive.txt`: mọi sample lấy được đúng membership split.
- `*_ignored_by_legacy_filter.txt`: sample mà phép cắt tên hiện tại bỏ qua.
- `legacy_rgb_filename_mapping_errors`: sample qua ID filter nhưng regex RGB hiện
  tại không đọc được; script báo lỗi, không sửa loader trong quá trình lọc dữ liệu.

Các index train/val/test mô phỏng **loader hiện tại**, không âm thầm đổi parser để
lấy thêm/bớt mẫu. Muốn sửa hạn chế loader là một thay đổi code riêng.

## 6. Input cần có

```text
D:/grasp-anything-pp/
├── grasp_label_positive.zip
├── grasp_instructions.zip
├── part_mask.zip
└── grasp_label_negative.zip       # tùy chọn

D:/grasp-anything/
├── scene_description.zip
├── mask.zip                       # tùy chọn
├── image.zip                      # hoặc truyền --images-dir
└── image/                         # ảnh đã giải nén, nếu có
```

Nếu ảnh còn ở hai phần ZIP ghép (`aa`, `ab`), phải ghép thành ZIP hợp lệ trước khi
truyền làm `image.zip`, hoặc dùng folder ảnh đã giải nén. Script không tải dữ liệu.
Chỉ xử lý một ZIP tại một thời điểm; Python vẫn phải đọc central directory của ZIP
lớn vào RAM. `tqdm` là tùy chọn để hiện tiến độ. Chỉ cần NumPy thêm nếu bật
`--ds-shuffle` để mô phỏng shuffle của trainer.

## 7. Lệnh dựng lại — bạn tự chạy

Từ thư mục repo, trong PowerShell:

```powershell
python tools/rebuild_grasp_anything_pp.py `
    --pp-archives-dir "D:/grasp-anything-pp" `
    --base-archives-dir "D:/grasp-anything" `
    --images-dir "D:/grasp-anything" `
    --output "data/grasp-anything-pp-rebuilt"
```

Để chỉ quét/lập báo cáo trước, thêm `--scan-only`. Sau đó chạy lại cùng lệnh không
có flag đó để trích xuất; marker cho phép hai bước dùng cùng output.
Để thay hardlink bằng copy, thêm `--link-mode copy`.

Các tham số kiểm toán train/val mặc định:

```text
--train-fraction 0.9 --ds-rotate 0.0 --random-seed 123
```

Thêm `--ds-shuffle` nếu bạn cũng bật tùy chọn đó khi train. Tham số trên chỉ sinh
index kiểm toán, không sửa `train_network.py`. Khi trainer chạy, nó tự chia lại
pool theo tham số của chính lệnh train.

Không xóa dataset cũ để chạy lại. Dùng output mới. Nếu chạy lại cùng input/output,
file giống nhau được xác minh và tái sử dụng. Nếu nguồn ZIP/split thay đổi hoặc
folder thuộc lựa chọn ID khác, script yêu cầu output mới để tránh trộn dữ liệu.

## 8. Output

```text
data/grasp-anything-pp-rebuilt/
├── _selection.json
├── scan_progress.json
├── report.json
├── missing_companions.jsonl
├── indices/
│   ├── train_seen_raw_positive.txt
│   ├── train_seen.txt
│   ├── train.txt
│   ├── val.txt
│   ├── test_seen.txt
│   ├── test_unseen.txt
│   └── *_ignored_by_legacy_filter.txt
├── selected/
│   ├── train_seen/
│   ├── test_seen/
│   └── test_unseen/
│       # từng nhóm có image/, scene_description/, grasp_label_positive/,
│       # grasp_instructions/, part_mask/, mask/ nếu có,
│       # annotation_inventory.csv, zip_inventory.csv và các danh sách thiếu
└── dataset/
    ├── grasp-anything/
    │   ├── image/
    │   ├── scene_description/
    │   └── mask/                   # nếu có
    └── grasp-anything-pp/
        ├── grasp_label_positive/
        ├── grasp_instructions/
        └── part_mask/
```

`indices/*.txt` là **tài liệu kiểm toán**; trainer không đọc chúng trực tiếp.
`dataset/` chỉ được tạo khi kiểm tra companion/khả năng map filename đạt;
`report.json` phải có `dataset_ready: true` trước khi dùng.
Nếu tạo layout bị gián đoạn, chạy lại cùng input/output để hoàn tất; không coi
folder `dataset/` tồn tại là bằng chứng quá trình đã hoàn thành.

`dataset_ready` xác nhận việc chọn ID/ghép file/CRC và layout, **không xác nhận
nội dung nhãn đúng về ngữ nghĩa**. Script không decode/đổi tensor grasp, không
kiểm chứng đơn vị tọa độ và không kiểm tra instruction có nhắc đúng part hay không.

## 9. Dùng dữ liệu mới với code train/test hiện tại

Trong notebook đã cấu hình môi trường GPU:

```python
from pathlib import Path
DATA_ROOT = REPO / "data/grasp-anything-pp-rebuilt/dataset"
DATASET_ROOT = DATA_ROOT / "grasp-anything"
ANNOTATION_ROOT = DATA_ROOT / "grasp-anything-pp"

!MPLBACKEND=Agg bash run_training.v1.sh \
    --dataset grasp-anywhere \
    --dataset-path "{DATASET_ROOT}" \
    --add-file-path "{ANNOTATION_ROOT}" \
    --text-source instruction
```

Model pretrained dùng mặc định hoặc truyền `--vision-model`, `--text-model` như
các lần chạy trước. Thêm `--segmentation` để đọc part mask; dùng cùng dataset và
text source cho thí nghiệm có/không segmentation.

Test seen chọn `--eval-split test_seen`; test unseen chọn `--eval-split test_unseen`
trong `evaluate_test.py`. `--seen 1` cũ chọn pool train seen, không phải held-out
seen test. `--dataset-path` và `--add-file-path` trỏ vào hai folder trong `dataset/`;
không truyền file JSON/TXT cho loader `grasp-anywhere`.

Khi đưa lên Kaggle, có thể chỉ mang thư mục `dataset/` để tránh đóng gói lặp payload
ở `selected/`. Giữ report/index/provenance riêng để đối chiếu thí nghiệm. Đường dẫn
trên Kaggle phải chỉnh theo vị trí bạn upload, không theo tên ổ D: trên Windows.
