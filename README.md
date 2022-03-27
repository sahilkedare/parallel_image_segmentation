## Parallel Image Segmentation
This repo implements the **quick shift** method to segment images on GPU. Memory access is the critical issue due to the size of image, variants of method are evaluated, yield to the conclusion that texture fetching is the most suitable solution. Compared with serial algorithm, this program performs a 130 to 300 times speed up, promises a real time super pixel computation on modest size.

### Quickshift Algorithm
<div style="text-align:center"><img src ="images/Algorithms.png" width=60% /></div>

### Evaluations
Refer to *Report.pdf* for more detailed numeric evaluation. Here are intuitive image outcomes,

> The Image Processing Girl, Lena

<div style="text-align:center"><img src ="images/lena/lena.png" width=40% /></div>
<div style="text-align:center">
<img src='images/lena/lena_2_10.png' width=30%>
<img src='images/lena/lena_2_20.png' width=30%>
</div>
<div style="text-align:center">
<img src='images/lena/lena_10_10.png' width=30%>
<img src='images/lena/lena_10_20.png' width=30%>
</div>
<p></p>

> The Mystery, Monalisa

<div style="text-align:center">
<img src ="images/monalisa/monalisa.png" width=40% />
</div>
<div style="text-align:center">
<img src='images/monalisa/monalisa_2_10.png' width=30%>
<img src='images/monalisa/monalisa_2_20.png' width=30%>
</div>
<div style="text-align:center">
<img src='images/monalisa/monalisa_10_10.png' width=30%>
<img src='images/monalisa/monalisa_10_20.png' width=30%>
</div>
