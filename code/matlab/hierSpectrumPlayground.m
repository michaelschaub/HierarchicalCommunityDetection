clc; clear; close all;
% construct hier. arrangement ala Tiago
p1 = [0.9, 0.11; 0.11, 0.9];
P = kron(kron(p1,p1),kron(p1,p1));
figure
imagesc(P)

% create partition vector
pvec = kron(1:16,ones(1,50));
H = transformPartitionVectorToHMatrix(pvec);



% big Prob matrix
Omega = H*P*H';
P = Omega > rand(size(Omega));
P = triu(P) + triu(P)' - 2*diag(diag(P));

P = H'*P*H;

D = diag(sum(P));
L = eye(size(D)) - 2*(D - P)/max(diag(D));
Lrohe = D^(-1/2)*P*D^(-1/2);

[u2,v2] = eig(L);
[a1,b] = sort(abs(diag(v2)),'descend');
u2 = u2(:,b);

[u1,v1] = eig(Lrohe);
[a2,b] = sort(abs(diag(v1)),'descend');
u1 = u1(:,b);


figure
plot(diag(v2))
figure
plot(u2(:,1:16))

figure
plot(diag(v1))
figure
plot(u1(:,1:16))

k=16;
x = zeros(1,k);
xx = zeros(1,k);
% [U,S,V] = svd(u1(:,1:k)');
[Q,R,perm] = qr(u1(:,1:k)','vector');
figure
plot(abs(Q'*u1(:,1:k)')')
for n_evals = 1:16
EV = u1(:,n_evals:end);
% figure
% plot(EV)

% part = kmeans(EV,n_evals);
[Q,R,perm] = qr(EV','vector');
[~,part] = max(abs(Q'*EV'));
figure;
plot(abs(Q'*EV')')
figure;
plot(part)
H3 = transformPartitionVectorToHMatrix(part);


error1 = (eye(length(EV)) - H3*((H3'*H3)^-1)*H3')*EV;
H3 = normc(H3);
error2 = (eye(length(EV))-EV*EV')*H3;
x(n_evals)  = norm(error1,'fro')/(n_evals-n_evals^2/length(EV));
xx(n_evals) = norm(error2,'fro')/(n_evals-n_evals^2/length(EV));
end
figure
plot(xx)
hold all
plot(x)
plot(x+xx)
