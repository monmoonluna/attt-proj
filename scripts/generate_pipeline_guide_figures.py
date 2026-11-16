#!/usr/bin/env python3
"""Generate explanatory diagrams and report charts for the Markdown study guide.

Read result tables from the guide; do not rerun ML experiments or modify Word.
Run: MPLCONFIGDIR=/tmp/nids-guide-mpl python3 scripts/generate_pipeline_guide_figures.py
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/nids-guide-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / 'docs/HuongDan_HieuPipeline_FeatureSelection_NIDS.md'
OUT = ROOT / 'docs/assets/pipeline_guide'
OUT.mkdir(parents=True, exist_ok=True)
TEXT = GUIDE.read_text()
NAVY, BLUE, GREEN, SLATE, PALE = '#0f172a', '#1e3a8a', '#166534', '#64748b', '#f8fafc'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'text.color':NAVY,
                     'axes.labelcolor':NAVY, 'axes.spines.top':False, 'axes.spines.right':False,
                     'figure.facecolor':'white', 'savefig.facecolor':'white'})
FIGURES = []

def save(fig, name):
    fig.savefig(OUT / (name+'.png'), dpi=170, bbox_inches='tight', pad_inches=.22)
    plt.close(fig)
    FIGURES.append(name)

def diagram(name, title, rows, note='Sơ đồ giải thích khái niệm; không phải số liệu thực nghiệm.'):
    """Equal-height rows, with parallel boxes in each row and arrows between rows."""
    h = len(rows)*1.45+1.3
    fig, ax = plt.subplots(figsize=(12,h))
    ax.set(xlim=(0,12), ylim=(0,h)); ax.axis('off')
    ax.text(0,h-.12,title,fontsize=17,weight='bold',va='top')
    for i,row in enumerate(rows):
        y=h-1.7-i*1.45
        gap=.35; w=(11.6-gap*(len(row)-1))/len(row)
        for j,label in enumerate(row):
            x=.2+j*(w+gap)
            ax.add_patch(Rectangle((x,y),w,1.05,facecolor=PALE,edgecolor=SLATE,lw=1))
            multiline = label.count('\n') >= 3
            ax.text(x+w/2,y+.525,label,ha='center',va='center',
                    fontsize=10 if multiline else 11,linespacing=1.2 if multiline else 1.55)
        if i<len(rows)-1:
            nxt = rows[i+1]
            nw = (11.6-gap*(len(nxt)-1))/len(nxt)
            pairs = ([(j,j) for j in range(len(row))] if len(row)==len(nxt)
                     else [(j,k) for j in range(len(row)) for k in range(len(nxt))])
            # The held-out test must not feed the fitness sample or learned parameters.
            if name in ('03-train-val-test','04-leakage') and i==1:
                pairs = [(0,0)]
            for j,k in pairs:
                ax.annotate('',xy=(.2+k*(nw+gap)+nw/2,y-.40),
                            xytext=(.2+j*(w+gap)+w/2,y-.03),
                            arrowprops={'arrowstyle':'->','color':SLATE,'lw':1})
    ax.text(.2,.05,note,fontsize=9,color=SLATE,va='bottom')
    save(fig,name)

def section(prefix):
    start=TEXT.index(prefix)
    end=TEXT.find('\n### ',start+len(prefix))
    return TEXT[start:end if end!=-1 else len(TEXT)]

def table(prefix, marker):
    lines=section(prefix).splitlines(); start=next(i for i,x in enumerate(lines) if x.startswith('|') and marker in x)
    records=[]
    for line in lines[start+2:]:
        if not line.startswith('|'): break
        records.append([x.strip() for x in line.strip('|').split('|')])
    return records

def number(s): return float(s.replace(',','.'))
def clean(ax, xlabel):
    ax.set_xlabel(xlabel); ax.grid(axis='x',color='#e2e8f0',lw=.7); ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)

def dots(name,title,labels,values,xlabel,limits=(0,1.07),note='Nguồn: các bảng kết quả trong báo cáo; một lần chạy, seed 42.'):
    fig,ax=plt.subplots(figsize=(11,max(4,len(labels)*.48+1.6)),layout='constrained')
    y=np.arange(len(labels))
    ax.hlines(y,limits[0],values,color='#cbd5e1',lw=2)
    ax.scatter(values,y,color=BLUE,s=48,zorder=3)
    for i,v in enumerate(values):
        label = f'{v:.2f}%' if 'Giảm chiều' in xlabel else f'{v:.4f}'
        ax.annotate(label,(v,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=10)
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(*limits)
    clean(ax,xlabel); ax.set_title(title,loc='left',fontsize=15,pad=18)
    fig.get_layout_engine().set(rect=(0,.07,1,1)); fig.text(.01,.015,note,fontsize=9,color=SLATE)
    save(fig,name)

def heatmap(name,title,data,rows,cols,note):
    a=np.asarray(data,dtype=float)
    fig,ax=plt.subplots(figsize=(10,max(4,len(rows)*.62+1.3)),layout='constrained')
    im=ax.imshow(a,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(len(cols)),cols); ax.set_yticks(range(len(rows)),rows)
    ax.tick_params(length=0)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j,i,f'{a[i,j]:.4f}',ha='center',va='center',color='white' if a[i,j]>.68 else NAVY,fontsize=11)
    ax.set_title(title,loc='left',fontsize=15,pad=18)
    fig.colorbar(im,ax=ax,label='Giá trị chỉ số' if 'Accuracy' in cols else 'F1-macro',shrink=.8)
    fig.get_layout_engine().set(rect=(0,.09,1,1));fig.text(.01,.015,note,fontsize=9,color=SLATE)
    save(fig,name)

# Conceptual illustrations follow the intended leakage-free protocol.
diagram('01-bai-toan','Bài toán: từ thông tin luồng đến nhãn dự đoán',[
 ['Các gói tin của một kết nối\nCông cụ thu thập gom thành luồng'],
 ['Một dòng đặc trưng X\nThời lượng · số gói · độ dài · cổng đích'],
 ['Chọn một tập cột nhỏ\nFeature Selection'],
 ['Bộ phân loại đã học\nDecision Tree / Random Forest / XGBoost'],
 ['Nhị phân\nNormal hoặc Attack','Đa lớp\nNormal, DoS, DDoS, Bots, ...']])
diagram('02-pipeline','Toàn bộ quy trình nghiên cứu',[
 ['CSV có nhãn → kiểm tra phân bố, lỗi nhãn và artifact'],
 ['Xác định train/test → học tiền xử lý trên train → biến đổi test'],
 ['Full\nGiữ tất cả','Filter\nMI','Wrapper\nGA / PSO','Hybrid\n1-GA / 1-PSO / 2'],
 ['Chốt tập cột → huấn luyện mô hình cuối trên toàn bộ train'],
 ['Đánh giá test → đổi bộ phân loại → ablation để phân tích']],
 'Quy trình nên áp dụng. Mã hiện tại có khác biệt về tiền xử lý; xem Mục 11.')
diagram('03-train-val-test','Vai trò của train, validation và test',[
 ['Dữ liệu ban đầu'],['Train: dùng để phát triển mô hình','Test: giữ riêng để đánh giá'],
 ['Mẫu tìm kiếm từ train\nCIC: 5.000 dòng; UNSW: toàn bộ train'],
 ['70%: học mô hình thử','30%: chấm fitness trên validation'],
 ['Chọn tập cột → học lại trên toàn bộ train → đánh giá trên test']],
 'Tỷ lệ 70/30 áp dụng cho dữ liệu tính fitness; không phải tỷ lệ train/test chung.')
diagram('04-leakage','Tiền xử lý: học trên train và áp dụng cho test',[
 ['Chia dữ liệu thô trước khi học tham số'],
 ['TRAIN\nHọc trung vị, min/max, bộ mã hóa, cột giữ lại','TEST\nKhông tự học lại các tham số'],
 ['Lưu tham số đã học từ train'],
 ['Biến đổi train','Dùng cùng tham số để biến đổi test'],
 ['Dữ liệu có cùng tập cột và cách biến đổi']],
 'Sơ đồ nguyên tắc đúng. Nhánh một tệp và điền thiếu của mã hiện tại cần sửa; xem Mục 11.')
diagram('05-confusion','Bốn tình huống khi phân loại nhị phân',[
 ['Thực tế là Attack','Thực tế là Normal'],
 ['TP: dự đoán Attack\nPhát hiện đúng\nFN: dự đoán Normal\nBỏ sót tấn công',
  'FP: dự đoán Attack\nCảnh báo nhầm\nTN: dự đoán Normal\nNhận đúng bình thường'],
 ['Recall = TP / (TP + FN)','FAR = FP / (FP + TN)']],
 'Quy ước Attack là lớp dương. Các ô giải thích định nghĩa, không biểu diễn số lượng.')
diagram('06-eda','EDA: từ quan sát đến câu hỏi cần kiểm tra',[
 ['Dữ liệu đầu vào'],
 ['Đếm nhãn\nCó đủ lớp không?','Thiếu / trùng\nCó lỗi dữ liệu không?','Tương quan / MI\nCột nào mạnh hoặc dư?'],
 ['Phát hiện kết quả đáng ngờ\nMột lớp duy nhất? Một cột gần như quyết định nhãn?'],
 ['Xem phân bố theo lớp → đối chiếu ý nghĩa cột → thử bỏ cột'],
 ['Giữ tín hiệu hợp lý hoặc xử lý lỗi / artifact']],
 'MI hoặc độ quan trọng cao chỉ là tín hiệu kiểm tra, không tự chứng minh rò rỉ.')
diagram('07-preprocess','Các thao tác tiền xử lý và đầu ra',[
 ['CSV → chuẩn hóa tên cột → xác định nhãn và train/test'],
 ['Inf → NaN → điền trung vị train','Xử lý trùng → xác định cột gần hằng trên train'],
 ['Loại định danh / nhãn khỏi X; CICDDoS2019 loại thêm Inbound'],
 ['Mã hóa cột chữ bằng ánh xạ train → chuẩn hóa bằng min/max train'],
 ['X_train, X_test, y_train, y_test + scaler, encoder, tên cột']],
 'Minh họa quy trình nên áp dụng; không xác nhận mã hiện tại đã đáp ứng mọi bước.')
diagram('08-mask','Vector bit biểu diễn tập đặc trưng',[
 ['Cột 1: thời lượng','Cột 2: số gói','Cột 3: cổng đích','Cột 4: IAT'],
 ['1: giữ','0: bỏ','1: giữ','0: bỏ'],
 ['Mô hình chỉ nhận cột 1 và cột 3'],
 ['Huấn luyện → chấm trên validation → fitness']],
 'Ví dụ minh họa 4 cột; không phải tập cột GA/PSO thực tế của báo cáo.')
diagram('09-filter','Filter-MI: xếp hạng từng cột',[
 ['Mẫu 20.000 dòng từ train'],
 ['Tính MI(cột 1, nhãn)','Tính MI(cột 2, nhãn)','Tính MI(cột ..., nhãn)'],
 ['Xếp hạng MI giảm dần → giữ khoảng 30% cột'],
 ['Tập cột đã chọn → huấn luyện bộ phân loại cuối']],
 'Chấm từng cột riêng; có thể giữ nhiều cột trùng thông tin và bỏ lỡ tương tác.')
diagram('10-ga','GA: tìm kiếm qua các thế hệ',[
 ['Khởi tạo 30 vector bit'],
 ['Chấm fitness: F1 validation và mức phạt số cột'],
 ['Chọn cha mẹ\nTournament 3','Lai ghép\nXác suất 0,8','Đột biến\nLật bit 1/n'],
 ['Giữ cá thể tốt nhất + tạo thế hệ tiếp theo'],
 ['Lặp 30 thế hệ → trả tập cột tốt nhất đã gặp']],
 'Fitness dùng DT trên validation. Không tối ưu trực tiếp theo F1 test.')
diagram('11-pso','PSO nhị phân và cơ chế duy trì đa dạng',[
 ['30 hạt; mỗi hạt có vector bit và vận tốc'],
 ['Chấm fitness → cập nhật pbest của hạt và gbest của bầy'],
 ['Cập nhật vận tốc → sigmoid → lấy vị trí bit mới'],
 ['Đột biến 15% hạt\nXác suất lật bit 3/n','5 vòng không tiến triển\nKhởi tạo lại 30% hạt kém'],
 ['Lặp 30 vòng → trả tập cột tốt nhất đã gặp']],
 'Cải thiện quá trình tìm kiếm không bảo đảm cải thiện F1 trên test.')
diagram('12-hybrid1','Hybrid 1: tiền lọc rồi tìm tập con',[
 ['Toàn bộ D đặc trưng'],
 ['MI giữ khoảng 50% cột → không gian ứng viên nhỏ hơn'],
 ['GA trên các cột ứng viên','PSO trên các cột ứng viên'],
 ['Fitness = 0,8 × F1-macro + 0,1 × F1 lớp hiếm − w_feat × k/n'],
 ['Chốt tập cột → huấn luyện mô hình cuối']],
 'n là số cột sau tiền lọc. Cột bị MI loại không thể được wrapper chọn lại.')
diagram('13-hybrid2','Hybrid 2: kết hợp hai cách chấm điểm',[
 ['Mẫu train cho lựa chọn đặc trưng'],
 ['MI\nQuan hệ từng cột với nhãn','XGBoost → SHAP\nĐóng góp trong mô hình'],
 ['Chuẩn hóa MI về [0, 1]','Chuẩn hóa mean|SHAP| về [0, 1]'],
 ['Điểm kết hợp = 0,5 × MI chuẩn hóa + 0,5 × SHAP chuẩn hóa'],
 ['Giữ khoảng 30% cột → dùng DT / RF / XGBoost đánh giá cuối']],
 'SHAP là giải thích mô hình, không phải bằng chứng nhân quả.')
diagram('14-ablation','Ablation: kiểm tra mô hình đang phụ thuộc vào gì',[
 ['Cùng dữ liệu, cùng bộ phân loại và thiết lập'],
 ['Đủ cột','Bỏ cột A','Bỏ cột B','Bỏ cả A và B'],
 ['Huấn luyện lại từng biến thể → so sánh F1'],
 ['Giảm mạnh: mô hình phụ thuộc cột; cần xem ý nghĩa và nguồn dữ liệu']],
 'Thay đổi F1 đo sự phụ thuộc trong phép thử; không tự xác định nguyên nhân.')

# Result charts: every numeric row is parsed from the Markdown tables.
for slug,prefix,title in [('cic','### 9.1.','CICIDS2017'),('unsw','### 9.2.','UNSW-NB15'),('ddos','### 9.3.','CICDDoS2019')]:
    r=table(prefix,'Giảm chiều'); labels=[x[0] for x in r]; f1=[number(x[3]) for x in r]
    dots(f'15-f1-{slug}',f'{title}: F1-macro nhị phân với Decision Tree',labels,f1,'F1-macro (thang đầy đủ 0–1)')
    dots(f'16-reduction-{slug}',f'{title}: tỷ lệ giảm số đặc trưng',labels,[number(x[2]) for x in r],
         'Giảm chiều (%)',(0,110),'Tính từ k/D trong báo cáo. Giảm chiều không phải tỷ lệ giảm số dòng.')
    # Avoid artificial midpoint for baseline timing interval: show the exact reported range.
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    for i,x in enumerate(r):
        if '–' in x[5]:
            lo,hi=map(number,x[5].split('–')); ax.plot([lo,hi],[i,i],color=SLATE,lw=6,solid_capstyle='butt')
            ax.text(hi*1.10,i,x[5],va='center')
        else:
            v=number(x[5]); ax.scatter(v,i,color=BLUE,s=48);ax.text(v*1.10,i,x[5],va='center')
    ax.set_xscale('log'); ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
    clean(ax,'T_train (giây, trục logarit; không gồm tìm kiếm đặc trưng)')
    ax.set_title(f'{title}: thời gian huấn luyện mô hình cuối',loc='left',pad=16)
    ax.set_xlim(right=max(number(r[0][5].split('–')[-1])*1.8, max(number(x[5]) for x in r[1:])*1.8))
    fig.get_layout_engine().set(rect=(0,.08,1,1));fig.text(.01,.015,'Đoạn xám là khoảng baseline được báo cáo; các điểm là thời gian từng phương án.',fontsize=9,color=SLATE)
    save(fig,f'17-train-time-{slug}')

r=table('### 9.4.','Bộ dữ liệu')
for dataset,slug in [('CICIDS2017','cic'),('UNSW-NB15','unsw')]:
    subset=[x for x in r if x[0]==dataset]
    heatmap(f'18-multi-{slug}',f'{dataset} đa lớp: Accuracy và F1-macro',
            [[number(x[3]),number(x[4])] for x in subset],[x[1] for x in subset],['F1-macro','Accuracy'],
            'Nguồn: Bảng 3.9. Accuracy cao có thể che khuất kết quả kém ở lớp hiếm.')
r=table('### 9.5.','Bộ dữ liệu')
for dataset,slug in [('CICIDS2017','cic'),('UNSW-NB15','unsw'),('CICDDoS2019','ddos')]:
    subset=[x for x in r if x[0]==dataset]
    heatmap(f'19-models-{slug}',f'{dataset}: cùng tập cột, đổi bộ phân loại',
            [[number(v) for v in x[3:6]] for x in subset],[x[1] for x in subset],['DT','RF','XGBoost'],
            'Nguồn: Bảng 3.10–3.11; nhị phân. XGBoost dùng thứ tự cột gốc.')
for prefix,name,title,source in [
 ('### 9.6.','20-ablation','CICDDoS2019: bỏ cổng đích và URG','Bảng 3.12; XGBoost dùng thứ tự gốc.'),
 ('### 9.7.','21-add-back','CICDDoS2019: thêm cột còn thiếu','Bảng 3.13; XGBoost dùng thứ tự JSON, cột thêm đặt cuối.')]:
    r=table(prefix,'F1 DT')
    heatmap(name,title,[[number(v) for v in x[2:5]] for x in r],[x[0] for x in r],['DT','RF','XGBoost'],source)

# Label distribution values as reported, not invented class counts.
fig,ax=plt.subplots(figsize=(11,4.4),layout='constrained')
normal=np.array([83.1,31.9,10.0]); attack=100-normal; y=np.arange(3)
ax.barh(y,normal,color=SLATE,label='Normal'); ax.barh(y,attack,left=normal,color=BLUE,label='Attack')
for i,(n,a) in enumerate(zip(normal,attack)):
    ax.text(n/2,i,f'{n:.1f}%',ha='center',va='center',color='white')
    ax.text(n+a/2,i,f'{a:.1f}%',ha='center',va='center',color='white')
ax.set_yticks(y,['CICIDS2017','UNSW-NB15 train','CICDDoS2019 train']);ax.invert_yaxis();ax.set_xlim(0,100)
ax.set_xlabel('Tỷ lệ mẫu (%)');ax.set_title('Phân bố nhãn nhị phân trước xử lý',loc='left',pad=20)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False)
save(fig,'22-label-balance')

r=table('### 9.2.','Giảm chiều')  # separate sensitivity table follows the narrative
lines=section('### 9.2.').splitlines(); idx=next(i for i,x in enumerate(lines) if 'w_feat |' in x)
sens=[]
for line in lines[idx+2:]:
    if not line.startswith('|'):break
    sens.append([v.strip() for v in line.strip('|').split('|')])
dots('23-penalty','Hybrid 1-GA: ảnh hưởng trọng số phạt số cột',
     [f'{x[0]} · w={x[1]} · {x[2]} cột' for x in sens],[number(x[3]) for x in sens],'F1-macro',
     note='Nguồn: Bảng 3.7. Không suy ra giảm trọng số luôn làm kết quả tốt hơn.')
dots('24-column-order','CICDDoS2019: XGBoost nhạy với thứ tự cột',
     ['Thứ tự JSON','Thứ tự cột gốc','Thứ tự đảo ngược'],[.8614,.8290,.8279],'F1-macro',
     note='Nguồn: Mục 3.3.2; cùng tập 10 cột Hybrid 1-PSO, chỉ thay thứ tự.')
dots('25-inbound','CICDDoS2019: baseline DT trước và sau loại Inbound',
     ['Còn Inbound · 68 cột','Đã loại Inbound · 67 cột'],[.8654,.9908],'F1-macro',
     note='Nguồn: Bảng 3.3. Kết quả sau xử lý cần đọc cùng phân tích cổng đích / URG.')
print(f'Generated {len(FIGURES)} figures in {OUT}')
