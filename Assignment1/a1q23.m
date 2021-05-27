% Question 2 and 3
h1=[0.24, 0.2, 0.16, 0.12, 0.08, 0.04, 0.12, 0.04];
h2=[ 0.22, 0.19, 0.16, 0.13, 0.11, 0.08, 0.05, 0.06];


prod=h1.*h2;
root=prod.^0.5;
b=sum(root);
ans=log(b);
ans=-1*log(b);
disp(sprintf("\nBhattacharya Distance is %f\n\n",ans));

division1=h1./h2;
division2=h2./h1;
product1=h1.*log(division1);
product2=h2.*log(division2);
a12=sum(product1);
a21=sum(product2);
disp(sprintf("KL Distance of 1 wrt 2 is %f",a12));
disp(sprintf("KL Distance is 2 wrt 1 is %f",a21));


% Question 3
hqminusht_trans=[0.5,0.5,-0.5,-0.25,-0.25]
hqminusht=hqminusht_trans'
a=[1,0.135,0.195,0.137,0.157;0.135,1,0.2,0.309,0.143;0.195,0.2,1,0.157,0.122;0.137,0.309,0.157,1,0.195;0.157,0.143,0.122,0.195,1]


sub2=hqminusht_trans*a;
ans2=sub2*hqminusht;
sub1=a*hqminusht;
ans1=hqminusht_trans*sub1;
disp(sprintf("Quadratic form Distance is %f",ans1));
