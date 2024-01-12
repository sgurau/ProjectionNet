load('matlab_matrix.mat')
y = labels=='AD';y = y(:,1)&y(:,2); y = y+1;
D = (D+D')/2;
A = -0.5*(D.^2);
n = size(D,1); H = eye(n) - ones(n)/n;
% K = sum(D,1)/n + sum(D,2)/n - sum(D(:))/n^2 - D;
K = H*A*H;
[U, d] = eig(K);
d = diag(d);
[d,idx]=sort(d,1,'descend');
U = U(:,idx(1:2));
d = d(1:2);
x = K*U*diag(sqrt(d))/sqrt(n);
s = 1:86;
% x = mdscale(D(s,s),2);
figure,scatter(x(:,1),x(:,2),50,y(s),'filled')
colormap(gca,'jet')
X = x;
%%
load 'matlab_matrix.mat' D labels
y = labels=='AD';y = y(:,1)&y(:,2); y = y+1;
D = (D+D')/2; D = D.^2;
s = [1:25,35:80];
ns = setdiff(1:86,s);
a = D(s,s);
n = length(s); l = length(ns);
H = eye(n) - ones(n)/n;

A = -0.5*a;
K = H*A*H;
[U, d] = eig(K);
d = diag(d);
[d,idx]=sort(d,1,'descend');
U = U(:,idx(1:2));
d = d(1:2);
x = K*U*diag(sqrt(d))/sqrt(n);
D0 = D(s,ns); D1 = D(s,s);
K0 = (ones(n)*D0/n +  D1*ones(n,l)/n - sum(D1,'all')/n^2 - D0)/2;
K0 = K0 - ones(n,n)*K0/n -K*ones(n,l)/n+sum(K,'all')/n^2;
x0 = K0'*U*diag(sqrt(d))/sqrt(n);
%%
figure
subplot(2,1,1)
scatter(x(:,1),x(:,2),50,y(s)), hold on
scatter(x0(:,1),x0(:,2),60,y(ns),'+')
colormap(gca,'jet')
subplot(2,1,2)
scatter(X(s,1),X(s,2),50,y(s),'filled'), hold on
scatter(X(ns,1),X(ns,2),50,y(ns),'filled','^')
colormap(gca,'jet')

%%
load 'matlab_matrix.mat' D labels
y = labels=='CN';y = y(:,1)&y(:,2); %y = 2-y;
cols = zeros(length(y),3);
D = (D+D')/2;
prop = 0.8; d=4;
tmp = find(y==1); cols(tmp,1) = 1;
msize = numel(tmp);
s = tmp(randperm(msize, floor(msize*prop)));
tmp = find(y==0); cols(tmp,2) = 1;
msize = numel(tmp);
s = [s;tmp(randperm(msize, floor(msize*prop)))];
% s = [1:25,35:80];
ns = setdiff(1:86,s);
a = D(s,s);
n = length(s); l = length(ns);
H = eye(n) - ones(n)/n;
opts = statset('MaxIter',1000);
P = projmdscale(D(s,s),d,'options',opts);
x = a*P;
X = D*projmdscale(D,d,'options',opts);

D0 = D(ns,s);
x0 = D0*P;

% figure
% subplot(2,1,1)
% scatter(x(:,1),x(:,2),50,cols(s,:)), hold on
% scatter(x0(:,1),x0(:,2),60,cols(ns,:),'+')
% subplot(2,1,2)
% scatter(X(s,1),X(s,2),50,cols(s,:),'filled'), hold on
% scatter(X(ns,1),X(ns,2),50,cols(ns,:),'filled','^')
% colormap(gca,'jet')


%
% % Load your data (replace this with your actual data loading code)
% load('your_data.mat');  % Make sure to replace 'your_data.mat' with your actual file
%
% % Split the data into features (X) and labels (y)
X = x;
yy = y(s);

% Set up cross-validation parameters
numFolds = 5;  % Number of folds for cross-validation
numRepeats = 50;  % Number of repeats for cross-validation
disp(['===================================Dimension = ',num2str(d),'==================================='])
fprintf('%d fold CV with %d repeat\n', numFolds, numRepeats);
% Define the hyperparameter search range
C_range = [0.01, 0.1, 1, 10, 100, 200, 400, 500, 800, 1000, 1500, 2000];  % Example values for C (adjust as needed)
gamma_range = [0.01, 0.1, 1, 10, 100, 200, 500];  % Example values for gamma (adjust as needed)

% Initialize variables to store results
bestAccuracy = 0;
bestf1Score = 0;
bestModel = [];
bestConfMat = zeros(2, 2);

% Perform cross-validation for hyperparameter tuning
cvp = [];
for repeat = 1:numRepeats
    cvp{repeat} = cvpartition(yy, 'KFold', numFolds);
end;
precision = zeros(numFolds*numRepeats, 1);
recall = zeros(numFolds*numRepeats, 1);
f1Scores = zeros(numFolds*numRepeats, 1);
precision = zeros(numFolds*numRepeats, 1);
innerAccuracy = zeros(numFolds*numRepeats, 1);
innerConfMat = zeros(2, 2, numFolds*numRepeats);
for C = C_range
    for gamma = gamma_range
        kk = 1;
        for repeat = 1:numRepeats
            for fold = 1:cvp{repeat}.NumTestSets
                trainIdx = training(cvp{repeat}, fold);
                testIdx = test(cvp{repeat}, fold);

                % Train SVM model
                svmModel = fitcsvm(X(trainIdx, :), yy(trainIdx), ...
                    'KernelFunction', 'RBF', 'BoxConstraint', C, 'KernelScale', gamma);

                % Predict on the test set
                yPred = predict(svmModel, X(testIdx, :));

                % Compute accuracy for this fold
                innerAccuracy(kk) = sum(yPred == yy(testIdx)) / numel(yy(testIdx));

                % Compute confusion matrix for this fold
                tmp = confusionmat(yy(testIdx), yPred);
                innerConfMat(:,:,kk) = tmp;
                precision(kk) = tmp(1, 1) / sum(tmp(1,:));
                recall(kk) = tmp(1, 1) / sum(tmp(:,1));
                f1Scores(kk) = 2 * (precision(kk) * recall(kk)) / (precision(kk) + recall(kk) + eps);
                kk = kk+1;
            end
        end
        % Compute mean confusion matrix across folds
        meanConfMat = mean(innerConfMat, 3);

        % Compute precision, recall, and F1-score
        meanprecision = mean(precision);
        meanrecall = mean(recall);
        meanf1Score = mean(f1Scores);
        stdf1Score = std(f1Scores);
        % Compute mean accuracy across folds
        meanAccuracy = mean(innerAccuracy);
        stdAccuracy = std(innerAccuracy);

        % Check if this hyperparameter combination gives a better accuracy
        if meanAccuracy > bestAccuracy
            %         if meanf1Score > bestf1Score
            disp(bestf1Score)
            bestf1Score = meanf1Score; bestf1ScoreSTD = stdf1Score;
            bestAccuracy = meanAccuracy; bestAccuracySTD = stdAccuracy;
            bestModel = fitcsvm(X, yy, 'KernelFunction', 'RBF', 'BoxConstraint', C, 'KernelScale', gamma);
            bestConfMat = mean(innerConfMat, 3);  % Use mean confusion matrix
            bestC = C; bestGamma = gamma;
        end
    end
end

% % Display the best hyperparameters
% bestC = bestModel.BoxConstraints;
% bestGamma = bestModel.KernelParameters.Scale;
fprintf('Best hyperparameters: C = %f, Gamma = %f\n', bestC, bestGamma);

% Display the best confusion matrix
disp('Best Confusion Matrix:');
disp(bestConfMat);

% Compute overall accuracy
% overallAccuracy = sum(diag(bestConfMat)) / sum(bestConfMat(:));
% fprintf('Overall Accuracy on new data: %.2f%%(+/-%.2f%%)\n', overallAccuracy * 100, bestAccuracySTD*100);
% overallAccuracy = sum(diag(bestConfMat)) / sum(bestConfMat(:));
fprintf('Overall F1 on new data: %f(+/-%f)\n', bestf1Score, bestf1ScoreSTD);

% Train the final model on the entire dataset using the best hyperparameters
finalModel = fitcsvm(X, yy, 'KernelFunction', 'RBF', 'BoxConstraint', bestC, 'KernelScale', bestGamma);

% Make predictions on new data
newData = x0;  % Replace this with the new data you want to predict
predictions = predict(finalModel, newData);

% Compute confusion matrix
confMat = confusionmat(y(ns), predictions);
precision = confMat(1, 1) / sum(confMat(1,:));
recall = confMat(1, 1) / sum(confMat(:,1));
f1Score = 2 * (precision * recall) / (precision + recall + eps);
% Display confusion matrix
disp('Confusion Matrix:');
disp(confMat);

% Compute accuracy
accuracy = sum(diag(confMat)) / sum(confMat(:));

fprintf('F1 on new data: %f\n', f1Score);
