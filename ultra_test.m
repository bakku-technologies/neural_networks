mfile = matfile("original_data/Pinch_and_Relax01_Ultrasound.mat");

imdata = mfile.ImgData;
imdata = imdata{1}(:,:,1,:);
lgI = log(imdata);

B=reshape(lgI, [39680,2000]);
csvwrite("original_data/pinch_relax_thevalues.csv", B');
C = reshape(B(:,1),[310,128]);
