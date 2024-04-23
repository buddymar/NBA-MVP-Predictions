# 🏀 Unlocking the MVP Formula: Analyzing Player and Team Metrics for NBA MVP Predictions
<br>

**Platform**: Jupyter Notebook | [Notebook via nbviewer](https://nbviewer.org/github/buddymar/NBA-MVP-Predictions/blob/main/NBA%20MVP.ipynb) | [Notebook via Github](https://github.com/buddymar/NBA-MVP-Predictions/blob/main/NBA%20MVP.ipynb)<br>
**Programming Language**: Python <br>
**Libraries**: Pandas, NumPy, sklearn, Matplotlib, Seaborn, BeautifulSoup, SHAP <br>
**Source Data**: scraped from [basketball-reference.com](https://www.basketball-reference.com/) <br>
<br>

**Table of Contents**
- [Introduction](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#-stage-0-problem-statement)
- [Data Scraping](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#-stage-1-data-preparation)
- [Data Integration](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#-stage-2-data-exploration)
- [Data Preprocessing](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#-stage-3-data-modeling-with-k-means-clustering)
	- [Data Cleaning](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
      - [Data Type](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
      - [Missing Values](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
      - [Duplicated Values](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
	- [Data Transformation](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
      - [Correcting Errors & Inconsistencies](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
      - [Creating New Columns](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
- [Exploratory Data Analysis](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#-stage-4-customer-personality-analysis)
  - [Player Attributes](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
  - [Players Basic Stats](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
  - [Players Advanced Stats](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
  - [Teams Stats](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
- [Predictive Modeling](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#-stage-5-business-recommendation)
  - [Modeling](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
  - [Model Prediction](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
  - [Model Interpretation](https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/blob/main/README.md#pre-processing)
<br>

---

## 📌 **Introduction**

<p align="center">
    <kbd> <img width="1000" alt="mvp banner" src="https://raw.githubusercontent.com/kudou88/NBA-MVP-Predictions/main/nba_mvp_banner.jpg"> </kbd> <br>
</p>

In the exhilarating world of professional basketball, the race for the NBA Most Valuable Player (MVP) award stands as a pinnacle of individual excellence and team leadership. Every season, NBA fans eagerly anticipate the unveiling of the MVP, an accolade bestowed upon the player deemed most instrumental to their team's success and exhibiting exceptional performance on the court.

The MVP selection process entails a comprehensive evaluation by a panel of sportswriters and broadcasters, who cast their votes based on a multitude of factors, including individual player statistics, team performance, impact on game outcomes, and intangible qualities such as leadership and sportsmanship. With the MVP race often culminating in heated debates and impassioned discussions among basketball enthusiasts, unraveling the underlying metrics that sway voters' decisions has become a compelling endeavor.

In this report, we undertake an in-depth exploration of the NBA MVP selection process. Utilizing a meticulously curated dataset obtained through web scraping, we delve into the intricate interplay of player attributes, fundamental and advanced statistical metrics, and team dynamics. Our objective is to unravel the complexities surrounding MVP success by scrutinizing the statistical profiles of past MVP recipients and dissecting the subtle intricacies of team performance. Through rigorous analysis, we endeavor to discern the pivotal factors that sway voters' decisions and forecast the leading contenders for the prestigious MVP accolade.

<br>

## 📌 **Data Scraping**

<p align="center">
    <kbd> <img width="1000" alt="bref" src="https://raw.githubusercontent.com/buddymar/NBA-MVP-Predictions/main/assets/BRef.jpg"> </kbd> <br>
</p>

The data utilized in this analysis and predictive modeling will be sourced from a website that hosts various basketball databases, [basketball-reference.com](https://www.basketball-reference.com/). The following data will be scraped, including:
- **MVP voting results:** Distribution of votes among players who received at least one vote each season.
- **Players basic stats:** All stats typically recorded in the official box score, aggregated per game for counting stats and per year for percentage stats.
- **Players advanced stats:** Analytical stats formulated and calculated using basic stats.
- **Team stats:** Stats concerning team performance and record for each season.

All this data is scraped for the period from 2001 to present.

<br>

## 📌 **Data Integration**

In this section, all datasets will be merged into a single dataset for analysis and prediction. Before proceeding, we will conduct feature selection to eliminate columns that are unnecessary, redundant, or contain similar information to columns that will be used in the analysis.

<br>

## 📂 **STAGE 3: Data Modeling with K-Means Clustering**
### Pre-processing
Sebelum melakukan data modeling, terdapat beberapa tahap pre-processing data yang perlu dilakukan yaitu:
- **Fitur yang tidak diperlukan** untuk model akan **dihapus** agar data lebih terfokus. 
- Fitur kategorikal akan di-**encoding** agar dapat diolah oleh algoritma machine learning. 
- Dilakukan **standardisasi** fitur untuk memastikan skala data seragam dan menghindari bias dalam model.<br>
<br>

### Modeling
Setelah pre-processing data selesai, tahap berikutnya adalah menggunakan metode **Principal Component Analysis (PCA)**. PCA digunakan untuk mengurangi dimensi data dengan mempertahankan informasi yang signifikan. Dengan mengurangi dimensi data, dapat mengoptimalkan kinerja model dan mengatasi masalah multicollinearity antara fitur. Selanjutnya, langkah penting dalam proses ini adalah menentukan jumlah cluster terbaik. Dalam analisis ini, **Distortion Score dan Elbow Method** digunakan untuk memilih jumlah cluster yang optimal. Berdasarkan hasil analisis, **jumlah cluster terbaik yang ditemukan adalah 4**.

<br>
<p align="center">
    <kbd> <img width="600" alt="distortion" src="https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/assets/115857221/176ddb7a-2357-49c0-8d06-1222973b0229"> </kbd> <br>
    Gambar 3 — Plot Distortion Scoce Elbow
</p>
<br>

Setelah menentukan jumlah cluster yang optimal, dilakukan **clustering menggunakan algoritma K-means**. Algoritma ini akan mengelompokkan data ke dalam cluster berdasarkan kesamaan fitur. Dengan melakukan clustering, dapat mengidentifikasi pola atau kelompok yang ada dalam data dan memahami karakteristik masing-masing cluster.

<br>
<p align="center">
    <kbd> <img width="600" alt="cluster" src="https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/assets/115857221/dda584d5-8519-4775-92c9-f7519bee8c6f"> </kbd> <br>
    Gambar 4 — Hasil Clustering menggunakan K-means
</p>
<br>

Dari plot hasil pemodelan dan pengelompokan data menggunakan metode clustering, terlihat bahwa **cluster-cluster yang terbentuk terpisah dengan baik** dan mengelompokkan data ke dalam kelompok yang berbeda-beda. Hal ini menunjukkan bahwa algoritma clustering yang digunakan berhasil dalam membedakan dan menggolongkan data berdasarkan karakteristik yang dimiliki.<br>
<br>

### Evaluation

<p align="center">
    <kbd> <img width="400" alt="score" src="https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/assets/115857221/1c6de7e4-aea2-4cfc-a6a8-402ab4a5b6c2"></kbd> <br>
    Gambar 5 — Hasil Evaluasi
</p>
<br>

Evaluasi hasil model menggunakan **Silhouette Score memberikan rekomendasi bahwa jumlah cluster terbaik adalah 4**. Hal ini didasarkan pada fakta bahwa nilai Silhouette Score pada jumlah cluster tersebut adalah yang tertinggi, yaitu 0.535. Silhouette Score merupakan metrik evaluasi yang menggambarkan seberapa baik objek-objek dalam satu cluster berada dalam kumpulan data mereka sendiri dibandingkan dengan cluster lainnya. Semakin tinggi nilai Silhouette Score, semakin baik cluster-cluster tersebut terpisah. <br>
<br>
<br>

---

## 📂 **STAGE 4: Customer Personality Analysis**
Customer Personality Analysis bertujuan untuk **memahami perbedaan dan kesamaan antara cluster-cluster tersebut, serta mengidentifikasi karakteristik unik yang mungkin dimiliki oleh setiap kelompok**. Dengan pemahaman yang lebih mendalam tentang karakteristik antar cluster, perusahaan dapat mengambil tindakan yang lebih tepat dan mengarahkan strategi bisnis yang lebih spesifik untuk setiap kelompok pelanggan.

<p align="center">
    <kbd> <img width="500" alt="income spending cluster" src="https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/assets/115857221/d85e981b-615f-4ace-a07b-ddcd32cb4457"></kbd> <br>
    Gambar 6 — Plot Pendapatan dan Total Pengeluaran Berdasarkan Cluster
</p>
<br>

Berdasarkan plot korelasi antara pendapatan (Income) dan total pengeluaran (Total Spending), terlihat bahwa terdapat pembentukan cluster atau kelompok yang dapat dibedakan. Dalam hal ini, **cluster 0 dan 3 cenderung berada dalam satu kelompok yang menunjukkan adanya persamaan dan perbedaan karakteristik di antara kedua cluster tersebut**. Ketika dua cluster berada dalam satu kelompok, hal ini mengindikasikan bahwa terdapat kemiripan atau keterkaitan dalam pola pendapatan dan pengeluaran di antara anggota-anggota cluster tersebut. Secara visual, terlihat bahwa **kedua cluster tersebut mungkin memiliki tingkat pendapatan dan pengeluaran yang relatif mirip atau memiliki tren yang serupa**.

<p align="center">
    <kbd> <img width="1000" alt="meancluster" src="https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/assets/115857221/2d66d306-a36d-46d2-a9f4-1f09f93abc1f"></kbd> <br>
    Gambar 6 — Plot Karakteristik Mayoritas/Rata-rata Total Transaksi, Pengeluaran, Pendapatan, Recency, dan Conversion Rate Berdasarkan Cluster
</p>
<br>

Berdasarkan hasil analisis yang lebih mendalam dapat diketahui karakteristik rata-rata/mayoritas dari setiap cluster berdasarkan pola transaksi pelanggan dan dapat dikelompokkan berdasarkan beberapa kategori.
- **Cluster 0**
    - Angka transaksi dan spending tertinggi yaitu mayoritas 25 transaksi dan Rp.1.116.000/bulan
    - Pendapatan cukup tinggi, mayoritas Rp.65.215.000/tahun
    - Conversion rate sedang, yaitu 4%
    - Kategori : **"*High-Transaction High-Spending Group*" - High Customer A** <br>
<br>

- **Cluster 1**
    - Angka transaksi dan spending terendah yaitu mayoritas hanya 7 transaksi dan Rp.58.000/bulan
    - Pendapatan terendah, mayoritas Rp.33.297.500/tahun
    - Conversion terendah, yaitu 1%
    - Kategori : **"*Low-Transaction Low-Spending Group*" - Low Customer** <br>
<br>
    
- **Cluster 2**
    - Angka transaksi dan spending cukup tinggi yaitu mayoritas 20 transaksi dan Rp.1.040.000/bulan
    - Pendapatan tertinggi, mayoritas Rp.71.488.000/tahun
    - Conversion rate tertinggi, yaitu 8%
    - Kategori : **"*High-Income High-Conversion Group*" - High Customer B** <br>
<br>

- **Cluster 3**
    - Angka transaksi dan spending sedang yaitu mayoritas 17 transaksi dan Rp.434.000/bulan
    - Pendapatan cukup sedang, mayoritas Rp.52.597.000/tahun
    - Conversion rate cukup sedang, yaitu 3%
    - Kategori : **"*Moderate-Transaction Moderate-Spending Group*" - Moderate Customer**<br>
<br>

Analisis distribusi beberapa fitur masing-masing cluster dilakukan juga dilakukan untuk mendapatkan wawasan yang lebih dalam. Melalui analisis ini, ditemukan beberapa insight menarik yang dapat memberikan pemahaman yang lebih baik tentang perilaku pengguna dalam setiap cluster, khususnya terkait kunjungan website dan respon terhadap campaign.

<p align="center">
    <kbd> <img width="1000" alt="distri clus" src="https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/assets/115857221/05707677-9a7e-47c2-b071-6f2d2c72c718"></kbd> <br>
    Gambar 7 — Plot Distribusi Berdasarkan Cluster
</p>
<br>

Berikut temuan yang menarik:
- **Low Customer (Cluster 1)** yang memiliki distribusi jumlah kunjungan website yang tinggi, namun memiliki total acceptance campaign yang rendah. Ini menunjukkan bahwa kelompok ini sangat **sering mengunjungi website perusahaan, tetapi tidak sepenuhnya menyadari atau tidak responsif terhadap campaign yang ditawarkan**. Mengingat kelompok ini memiliki populasi yang paling banyak, perusahaan perlu mengembangkan strategi yang tepat untuk menarik perhatian dan meningkatkan keterlibatan mereka. 
- Cluster yang **paling banyak merespon campaign adalah High Customer A (Cluster 0)** dengan tingkat konversi yang sedang. Ini menunjukkan bahwa mayoritas pelanggan dalam kelompok ini sangat responsif terhadap campaign yang ditawarkan oleh perusahaan. Hal ini dapat menjadi kesempatan yang baik untuk meningkatkan interaksi dan pembelian dari kelompok ini dengan meluncurkan campaign yang lebih menarik dan relevan sesuai dengan preferensi mereka.
- **High Customer B (Cluster 2)**, mayoritas pelanggannya tidak terlalu sering mengunjungi website perusahaan, namun memiliki distribusi konversi rate yang lebih tinggi dengan respon campaign yang sedang. Fenomena ini menunjukkan bahwa kelompok ini **memiliki kecenderungan pengeluaran yang tinggi dan cenderung merespons positif terhadap campaign yang ditawarkan, meskipun mereka tidak begitu aktif dalam kunjungan ke website**. Perusahaan dapat memanfaatkan informasi ini dengan mengoptimalkan saluran komunikasi lain seperti email, media sosial, atau platform online lainnya untuk efektif menjangkau kelompok ini.

<br>
<p align="center">
    <kbd> <img width="600" alt="percentage" src="https://github.com/faizns/Predict-Customer-Personality-to-Boost-Marketing-Campaign/assets/115857221/d7c0d96e-3d15-4d59-9ffc-b7afed8786a0"></kbd> <br>
    Gambar 8 — Plot Presentase Populasi Cluster 
</p>
<br>

Berdasarkan persentase populasi masing-masing cluster, ditemukan bahwa **50.22% dari keseluruhan pelanggan termasuk dalam kelompok Low Customer (Cluster 1)**. Meskipun kelompok ini memiliki angka transaksi dan pengeluaran yang rendah, namun karena populasi mereka yang besar. Perusahaan dapat fokus untuk menarik perhatian mereka. Sedangkan populasi **High Customer A (Cluster 0) dan B (Cluster 2) cenderung rendah**, namun memiliki potensi transaksi dan spending yang tinggi. Perusahaan dapat mempertimbangkan strategi pemasaran yang lebih personal dan eksklusif untuk menarik minat mereka.

<br>
<br>

---

## 📂 **STAGE 5: Business Recommendation**

Berdasarkan analisis yang telah dilakukan, dapat diidentifikasi personalitas atau karakteristik pelanggan berdasarkan cluster yang terbentuk. Mengetahui karakteristik ini sangat berharga dalam merancang strategi pemasaran yang lebih efektif. Dengan memahami preferensi, kebutuhan, dan perilaku konsumen dalam setiap cluster, perusahaan dapat menghasilkan campaign yang lebih relevan dan menarik bagi setiap kelompok pelanggan.

### High Customer A
Summary:
- Populasi 12.61%.
- High-Transaction High-Spending Group.
- Paling responsif terhadap campaign, dengan tingkat kunjungan website dan konversi ke pembelian sedang.

Rekomendasi:
- Mengingat kelompok High Customer A cenderung memiliki total transaksi dan total spending yang tinggi, perusahaan dapat memberikan **penawaran khusus dan insentif tambahan untuk mendorong pelanggan melakukan pembelian secara terus-menerus**. Perusahaan dapat menerapkan program diskon eksklusif, hadiah loyalitas, atau akses ke produk atau layanan khusus untuk kelompok ini.
- Perusahaan dapat meningkatkan **kualitas pengalaman pengguna dalam berselancar di website, mengingat tingkat kunjungan website yang sedang**. Perusahaan dapat memastikan tampilan yang menarik, customer journey yang efisien, dan lain sebagainya.
- Mengingat kelompok High Customer A sangat responsif terhadap campaign, memanfaatkan kepuasan mereka dengan memperkenalkan **program referral** dapat menjadi strategi yang efektif. Memberikan insentif kepada pelanggan untuk merekomendasikan produk atau layanan perusahaan kepada teman dan keluarga dapat membantu dalam memperluas jangkauan dan memperoleh pelanggan baru.
- Perusaan dapat **mingirimkan pesan yang dipersonalisasikan** seperti info promo atau diskon **berdasarkan preferensi** kelompok ini. Hal ini dilakukan untuk menjaga loyalitas pelanggan. <br>
<br>


### High Customer B
Summary :
- Populasi 13.42%.
- High-Income High-Conversion Group.
- Sama seperti High Customer B dalam segi income dan total spending, namun memiliki income paling tinggi.
- Tingkat konversi paling tinggi, respon terhadap campaign relatif sedang, kurang mengunjungi website secara aktif. 

Rekomendasi:
- Sama halnya dengan High Customer A, perusahaan dapat **memberikan penawaran khusus** seperti diskon, program loyalti, dan sebagainya agar pelanggan selalu tertarik untuk berbelanja terus menerus.
- Mengingat kelompok ini kurang aktif dalam kunjungan website, perusahaan dapat memanfaatkan **saluran komunikasi alternatif untuk campaign** seperti email, pesan teks, atau media sosial. Hal ini dapat membantu meningkatkan interaksi dan kesadaran pelanggan.
- Untuk meningkatkan respon pelanggan terhadap campaign, perusahaan dapat **memberikan campaign-campaign yang tertarget sesuai dengan preverensi dan kebutuhan pelanggan**.
- Mengingat pelanggan dalam kelompok High Customer B memiliki tingkat konversi yang tinggi, perusahaan dapat mempertimbangkan untuk meluncurkan **program loyalitas** yang memberikan insentif tambahan, penghargaan khusus, atau akses ke acara atau produk eksklusif dapat memperkuat loyalitas pelanggan. <br>
<br>


### Moderate Customer
Summmary:
- Populasi 23.75%.
- Moderate-Transaction Moderate-Spending Group
- Tingkat konversi, kunjungan website dan respon terhadap campaign relatif sedang.

Rekomendasi:
- Perusahaan dapat memberikan **penawaran khusus** dan diskon untuk mendorong pembelian lebih lanjut. Hal ini dapat memberikan insentif tambahan kepada pelanggan dalam kelompok ini untuk memilih produk atau layanan perusahaan dibandingkan dengan pesaing.
- Perusahaan dapat **mengirim pesan yang relevan dan menarik** kepada pelanggan untuk melakukan transaksi.
- **Memastikan pengalaman pengguna yang baik saat mengunjungi website** atau berinteraksi dengan produk atau layanan perusahaan.
- Membangun **program hadiah atau loyalitas** dapat membantu memperkuat keterikatan pelanggan. Seperti dengan memberikan poin, penghargaan, atau manfaat khusus kepada pelanggan setia, perusahaan dapat mendorong mereka untuk terus memilih dan membeli produk atau layanan perusahaan. <br>
<br>


### Low Customer
Summary:
- Populasi 50.22%, pelanggan didominasi oleh kategori ini.
- Low-Transaction Low-Spending Group.
- Tingkat konversi paling rendah, cenderung tidak merespon campaign, namun kategori ini paling sering mengunjungi website.

Rekomendasi:
- Mengingat kelompok Low Customer sering mengunjungi website, perusahaan dapat **memanfaatkan informasi kunjungan website untuk menyajikan konten yang personalisasi dan penawaran khusus yang sesuai dengan minat dan preferensi mereka**.
- Perusahaan dapat **melakukan retargeting campaign** dengan mengingatkan pelanggan dalam kelompok ini tentang produk atau layanan yang mereka telah kunjungi di website. Dengan menampilkan iklan yang disesuaikan di berbagai platform digital yang mereka gunakan, perusahaan dapat membangun kesadaran dan mendorong mereka untuk melanjutkan proses pembelian.
- Mengingat kelompok Low Customer memiliki tingkat konversi yang rendah dan cenderung tidak merespon campaign dengan baik, perusahaan dapat **menggunakan strategi konten yang lebih fokus pada edukasi dan informasi (softselling)**. Memberikan konten yang memberikan nilai tambah, memberikan solusi untuk masalah atau kebutuhan pelanggan, dan membantu mereka membuat keputusan yang lebih informatif dapat meningkatkan keterlibatan dan kepercayaan pelanggan dalam kelompok ini.
