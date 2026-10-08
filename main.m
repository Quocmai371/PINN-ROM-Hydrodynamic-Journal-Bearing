clear
clc
close all

%% SETTINGS

build_pod = false;
train_model = false;
run_single_case = false;
run_100_cases = false;
run_rotor = true;
run_stability = true;

bearing_method = "BOTH";      % "FVM", "PINN", "BOTH"

%% BEARING

model.R = 45e-3;
model.L = 60e-3;
model.c = 0.25e-3;
model.mu = 0.015;

model.D = 2*model.R;
model.lambda = model.L/model.D;
model.psi = model.c/model.R;
model.k = model.R/model.L;

model.nx = 80;
model.nz = 60;

model.epsMin = 0.01;
model.epsMax = 0.97;  % overwritten by the loaded POD range below

%% POD

model.snapshotGlobalPoints = 600;
model.snapshotHighPoints = 800;
model.snapshotLowPoints = 500;
model.snapshotHighStart = 0.80;
model.snapshotLowEnd = 0.25;

model.podEnergy = 0.999999;
model.podMinModes = 12;
model.podMaxModes = 40;
model.podP95Target = 0.01;

model.bcFourierModes = 4;
model.bcAxialModes = 4;
model.bcModeScale = 1.0;

%% PINN

model.numEpochs = 400000;
model.batchSize = 128;
model.networkWidth = 256;
model.lambdaMin = 0.5;
model.lambdaMax = 2.0;

model.fullResidualRows = 8;
model.galerkinWeight = 1.0;
model.fullResidualWeight = 1.0;
model.pdeWeight = 1.0;
model.bcWeight = 1.0;
model.axialBCWeight = 1.0;
model.periodicQBCWeight = 1.0;
model.periodicQxBCWeight = 1.0;

model.highEpsWeight = 6.0;
model.lowEpsWeight = 5.0;
model.gradClip = 2.0;

% LEARNING RATE DECAY

model.lrStart = 1e-3;

% Phase 1: decay faster
model.lrDecayRate1 = 0.90;
model.lrDecayPeriod1 = 1900;

% Switch near LR ~ 2e-5
model.lrDecaySwitch = 70000;

% Phase 2: decay slower for refinement
model.lrDecayRate2 = 0.97;
model.lrDecayPeriod2 = 3000;

model.lrMin = 2e-7;

model.validationEvery = 100;
model.printEvery = 1000;

%% 100 CASES

numCases = 100;
validationSeed = 21;

% model.epsilonTauMin = 1e-5;
% model.epsilonTauMax = 2e-2;
% model.gammaTauMin = 0;
% model.gammaTauMax = 1;
% model.omegaMin = 100;
% model.omegaMax = 1400;

%% RIGID ROTOR 2-DOF

model.mass = 20;
model.gravity = 9.81;
model.numberBearings = 2;

model.speedMode = "RUNUP";    % "FIXED" or "RUNUP"
model.fixedOmega = 300;
model.omega0 = 100;
model.alpha = 60;

model.dt = 1e-4;
model.timeEnd = 20;

model.z0 = 1e-6*randn(4,1);

% Same unbalance form as the reference rotor code.
model.unbalanceEccentricity = 0.001*6.3/250;

% External load [Qx; Qy], positive y downward.
model.Q = [0;0];

%% STABILITY

model.stabilityOmega = (100:2:1600).';
model.DXratio = 2e-4;
model.DVratio = 1e-7;
model.eigFile = 'eigA_results.mat';
model.plotEigenvalues = true;
%% POD AND PINN

podFile = 'pod_model.mat';
netFile = 'pinn_model.mat';

if build_pod
    pod = PODGalerkin(model,podFile,true);
elseif isfile(podFile)
    load(podFile,'pod')
else
    error('pod_model.mat not found. Set train_model = true to train the model.')
end

model.epsMin = pod.epsMin;
model.epsMax = pod.epsMax;

if train_model
    net = TrainPINN(pod,model,netFile);
elseif isfile(netFile)
    load(netFile,'net')
else
    error('pinn_model.mat not found. Set train_model = true to train the model.')
end
%% PLOT TRAINING HISTORY

load('pinn_model_history.mat','history')

figure

semilogy(history.epoch,history.total)
hold on

semilogy(history.epoch,history.pde)
semilogy(history.epoch,history.gal)
semilogy(history.epoch,history.full)
semilogy(history.epoch,history.bc)

xlabel('Epoch')
ylabel('Loss')

title('POD-Galerkin ROM PINN Training Loss')

legend('Total','PDE','Galerkin','Full','BC', ...
    'Location','best')

grid on

%%
% % SINGLE CASE SETTINGS
% % =========================================================
% 
% study = "EPSILON";   % "EPSILON", "SPEED", "CLEARANCE", "SQUEEZE", "ATTITUDE"
% 
% gamma = 0;
% zCut = 0.50;
% 
% c0 = model.c;
% psi0 = model.psi;
% 
% switch study
% 
%     case "EPSILON"
%         values = [0.30 0.50 0.70 0.90];
%         epsilon0 = 0.50;
%         omega0 = 314;
%         epsilonTau0 = 0;
%         gammaTau0 = 0;
%         model.c = 0.25e-3;
%         repIndex = 2;
%         xLabel = 'Eccentricity Ratio, \epsilon';
% 
%     case "SPEED"
%         values = [3000 4500 6000 7500];
%         epsilon0 = 0.50;
%         omega0 = 628.32;
%         epsilonTau0 = 0;
%         gammaTau0 = 0;
%         model.c = 0.25e-3;
%         repIndex = 3;
%         xLabel = 'Rotational Speed [rpm]';
% 
%     case "CLEARANCE"
%         values = [0.15 0.20 0.25 0.30];
%         epsilon0 = 0.50;
%         omega0 = 314.16;
%         epsilonTau0 = 0;
%         gammaTau0 = 0;
%         repIndex = 1;
%         xLabel = 'Radial Clearance, c [mm]';
% 
%     case "SQUEEZE"
%         values = [0 2 4 6 10]*1e-4;
%         epsilon0 = 0.70;
%         omega0 = 628.32;
%         epsilonTau0 = 0;
%         gammaTau0 = 0;
%         model.c = 0.25e-3;
%         repIndex = 1;
%         xLabel = '\dot{e}/(\Omega R)';
% 
%     case "ATTITUDE"
%         values = [0 0.2 0.4 0.5 0.6 0.8];
%         epsilon0 = 0.70;
%         omega0 = 628.32;
%         epsilonTau0 = 0;
%         gammaTau0 = 0;
%         model.c = 0.25e-3;
%         repIndex = 1;
%         xLabel = '\dot{\gamma}/\Omega';
% 
% end
% 
% model.psi = model.c/model.R;
% 
% 
% %% =========================================================
% % COMMON GRID
% % =========================================================
% 
% xbar = linspace(0,1,model.nx);
% zbar = linspace(0,1,model.nz);
% 
% [XBAR,ZBAR] = meshgrid(xbar,zbar);
% 
% xFine = linspace(0,1,320);
% zFine = linspace(0,1,300);
% 
% [XFINE,ZFINE] = meshgrid(xFine,zFine);
% 
% thetaDeg = 360*xbar;
% 
% [~,izMid] = min(abs(zbar-zCut));
% 
% nCases = length(values);
% 
% profileFVM = zeros(nCases,model.nx);
% profilePINN = zeros(nCases,model.nx);
% 
% pMaxFVM = zeros(nCases,1);
% pMaxPINN = zeros(nCases,1);
% 
% pressureError = zeros(nCases,1);
% forceError = zeros(nCases,1);
% 
% errorFineAll = zeros(length(zFine),length(xFine),nCases);
% maxErrorAll = zeros(nCases,1);
% 
% pFVMall = zeros(model.nz,model.nx,nCases);
% pPINNall = zeros(model.nz,model.nx,nCases);
% 
% 
% %% =========================================================
% % COMMON CALCULATION
% % =========================================================
% 
% for i = 1:nCases
% 
%     epsilon = epsilon0;
%     omega = omega0;
%     epsilonTau = epsilonTau0;
%     gammaTau = gammaTau0;
% 
%     switch study
% 
%         case "EPSILON"
%             epsilon = values(i);
% 
%         case "SPEED"
%             omega = values(i)*2*pi/60;
% 
%         case "CLEARANCE"
%             model.c = values(i)*1e-3;
%             model.psi = model.c/model.R;
% 
%         case "SQUEEZE"
%             epsilonTau = values(i)/model.psi;
% 
%         case "ATTITUDE"
%             gammaTau = values(i);
% 
%     end
% 
%     qFVM = ReynoldsFVM(model,epsilon,gamma,epsilonTau,gammaTau);
%     qPINN = ReynoldsPINN(net,pod,model,epsilon,gamma,epsilonTau,gammaTau);
% 
%     pScale = model.mu*omega*(model.R/model.c)^2;
% 
%     pFVM = qFVM*pScale/1e6;
%     pPINN = qPINN*pScale/1e6;
% 
%     pFVMall(:,:,i) = pFVM;
%     pPINNall(:,:,i) = pPINN;
% 
%     profileFVM(i,:) = pFVM(izMid,:);
%     profilePINN(i,:) = pPINN(izMid,:);
% 
%     pMaxFVM(i) = max(pFVM(:));
%     pMaxPINN(i) = max(pPINN(:));
% 
%     pError = abs(pPINN-pFVM);
% 
%     maxErrorAll(i) = max(pError(:));
%     errorFineAll(:,:,i) = interp2(XBAR,ZBAR,pError,XFINE,ZFINE,'cubic');
% 
%     pressureError(i) = 100*norm(qPINN(:)-qFVM(:))/max(norm(qFVM(:)),1e-14);
% 
%     FFVM = forceFromPressure(qFVM,omega,model);
%     FPINN = forceFromPressure(qPINN,omega,model);
% 
%     forceError(i) = 100*norm(FPINN-FFVM)/max(norm(FFVM),1e-14);
% 
% end
% 
% 
% %% =========================================================
% % REPRESENTATIVE DATA
% % =========================================================
% 
% pFVMrep = pFVMall(:,:,repIndex);
% pPINNrep = pPINNall(:,:,repIndex);
% 
% pFVMFine = interp2(XBAR,ZBAR,pFVMrep,XFINE,ZFINE,'cubic');
% pPINNFine = interp2(XBAR,ZBAR,pPINNrep,XFINE,ZFINE,'cubic');
% 
% pErrorFine = abs(pPINNFine-pFVMFine);
% 
% pMin = min([pFVMrep(:);pPINNrep(:)]);
% pMax = max([pFVMrep(:);pPINNrep(:)]);
% 
% fprintf('\nPressure error = %.4f %%\n',pressureError(repIndex))
% fprintf('Force error    = %.4f %%\n',forceError(repIndex))
% 
% 
% %% =========================================================
% % PLOT 1 - REPRESENTATIVE PRESSURE
% % =========================================================
% 
% figure('Position',[80 80 1250 800])
% 
% subplot(2,2,1)
% surf(XFINE,ZFINE,pFVMFine,'EdgeColor','none')
% xlabel('\bar{x}')
% ylabel('\bar{z}')
% zlabel('Pressure [MPa]')
% title(sprintf('FVM, p_{max} = %.4f MPa',pMaxFVM(repIndex)))
% clim([pMin pMax])
% colorbar
% view(45,30)
% grid on
% 
% subplot(2,2,2)
% surf(XFINE,ZFINE,pPINNFine,'EdgeColor','none')
% xlabel('\bar{x}')
% ylabel('\bar{z}')
% zlabel('Pressure [MPa]')
% title(sprintf('PINN, p_{max} = %.4f MPa',pMaxPINN(repIndex)))
% clim([pMin pMax])
% colorbar
% view(45,30)
% grid on
% 
% subplot(2,2,3)
% contourf(XFINE,ZFINE,pErrorFine,30,'LineColor','none')
% xlabel('\bar{x}')
% ylabel('\bar{z}')
% title(sprintf('Absolute Error, max = %.4f MPa',maxErrorAll(repIndex)))
% colorbar
% grid on
% 
% subplot(2,2,4)
% plot(thetaDeg,profileFVM(repIndex,:),'LineWidth',1.6)
% hold on
% plot(thetaDeg,profilePINN(repIndex,:),'--','LineWidth',1.6)
% xlabel('Angular Coordinate, \theta [deg]')
% ylabel('Pressure [MPa]')
% legend('FVM','PINN','Location','best')
% xlim([0 360])
% grid on
% 
% 
% %% =========================================================
% % PLOT 2 - ALL PRESSURE PROFILES
% % =========================================================
% 
% figure
% 
% plot(thetaDeg,profileFVM.','LineWidth',1.4)
% hold on
% plot(thetaDeg,profilePINN.','--','LineWidth',1.4)
% 
% xlabel('Angular Coordinate, \theta [deg]')
% ylabel('Pressure [MPa]')
% xlim([0 360])
% grid on
% box on
% 
% 
% %% =========================================================
% % PLOT 3 - MAXIMUM PRESSURE
% % =========================================================
% 
% figure
% 
% plot(values,pMaxFVM,'-o','LineWidth',1.6,'MarkerSize',7)
% hold on
% plot(values,pMaxPINN,'--s','LineWidth',1.6,'MarkerSize',7)
% 
% xlabel(xLabel)
% ylabel('Maximum Pressure [MPa]')
% legend('FVM','PINN','Location','best')
% xticks(values)
% grid on
% box on
% 
% 
% %% =========================================================
% % RESTORE MODEL
% % =========================================================
% 
% model.c = c0;
% model.psi = psi0;
%% ROTOR

resultFVM = [];
resultPINN = [];

timeFVM = NaN;
timePINN = NaN;

if run_rotor

    if bearing_method == "FVM" || bearing_method == "BOTH"
        tStart = tic;
        resultFVM = RotorAnalysis("TRANSIENT","FVM",net,pod,model);
        timeFVM = toc(tStart);
        fprintf('FVM rotor time = %.2f hr\n',timeFVM/3600)
    end

    if bearing_method == "PINN" || bearing_method == "BOTH"
        tStart = tic;
        resultPINN = RotorAnalysis("TRANSIENT","PINN",net,pod,model);
        timePINN = toc(tStart);
        fprintf('PINN rotor time = %.2f hr\n',timePINN/3600)
    end

    TF=resultFVM.Time_s;
    xF=resultFVM.X_over_c;
    yF=resultFVM.Y_over_c;
    eF=resultFVM.Epsilon;
    FbxF=resultFVM.BearingFx_N;
    FbyF=resultFVM.BearingFy_N;

    TP=resultPINN.Time_s;
    xP=resultPINN.X_over_c;
    yP=resultPINN.Y_over_c;
    eP=resultPINN.Epsilon;
    FbxP=resultPINN.BearingFx_N;
    FbyP=resultPINN.BearingFy_N;

    save('rotor_results.mat', ...
        'TF','xF','yF','eF','FbxF','FbyF', ...
        'TP','xP','yP','eP','FbxP','FbyP', ...
        'timeFVM','timePINN')

elseif isfile('rotor_results.mat')

    load('rotor_results.mat', ...
        'TF','xF','yF','eF','FbxF','FbyF', ...
        'TP','xP','yP','eP','FbxP','FbyP', ...
        'timeFVM','timePINN')

end
%% STABILITY

stabilityFVM = [];
stabilityPINN = [];

timeStabilityFVM = NaN;
timeStabilityPINN = NaN;

if run_stability

    if bearing_method == "FVM" || bearing_method == "BOTH"
        tStart = tic;
        stabilityFVM = RotorAnalysis("STABILITY","FVM",net,pod,model);
        timeStabilityFVM = toc(tStart);
    end

    if bearing_method == "PINN" || bearing_method == "BOTH"
        tStart = tic;
        stabilityPINN = RotorAnalysis("STABILITY","PINN",net,pod,model);
        timeStabilityPINN = toc(tStart);
    end

    save('stability_results.mat', ...
        'stabilityFVM','stabilityPINN', ...
        'timeStabilityFVM','timeStabilityPINN')

elseif isfile('stability_results.mat')

    load('stability_results.mat', ...
        'stabilityFVM','stabilityPINN', ...
        'timeStabilityFVM','timeStabilityPINN')

end
%% LOAD RESULTS
load('pinn_model.mat')
load('pinn_model_history.mat')
load('pod_model.mat')
load('rotor_results.mat')
load('stability_results.mat')
%% ROTOR RESPONSE

figure

subplot(2,2,1)
plot(TF,xF); hold on
plot(TP,xP,'--')
xlabel('Time [s]'); ylabel('x/c'); title('Horizontal Response')
legend('FVM','PINN','Location','best'); grid on

subplot(2,2,2)
plot(TF,yF); hold on
plot(TP,yP,'--')
xlabel('Time [s]'); ylabel('y/c'); title('Vertical Response')
legend('FVM','PINN','Location','best'); grid on

subplot(2,2,3)
plot(xF,yF); hold on
plot(xP,yP,'--')
xlabel('x/c'); ylabel('y/c')
axis equal
legend('FVM','PINN','Location','best'); grid on

subplot(2,2,4)
plot(TF,eF); hold on
plot(TP,eP,'--')
xlabel('Time [s]'); ylabel('\epsilon')
legend('FVM','PINN','Location','best'); grid on


%% HYDRODYNAMIC BEARING FORCE

figure

subplot(2,1,1)
plot(TF,FbxF); hold on
plot(TP,FbxP,'--')
xlabel('Time [s]'); ylabel('F_{b,x} [N]')
legend('FVM','PINN','Location','best'); grid on

subplot(2,1,2)
plot(TF,FbyF); hold on
plot(TP,FbyP,'--')
xlabel('Time [s]'); ylabel('F_{b,y} [N]')
legend('FVM','PINN','Location','best'); grid on
%% RESULTANT HYDRODYNAMIC BEARING FORCE

FbF = sqrt(FbxF.^2 + FbyF.^2);
FbP = sqrt(FbxP.^2 + FbyP.^2);

figure

plot(TF,FbF,'LineWidth',1.2)
hold on
plot(TP,FbP,'--','LineWidth',1.2)

xlabel('Time [s]')
ylabel('|F_b| [N]')

legend('FVM','PINN','Location','best')

grid on
box on

%% STABILITY DATA

rpmF = stabilityFVM.data.RPM(:);
rpmP = stabilityPINN.data.RPM(:);

eigF = stabilityFVM.lambda;
eigP = stabilityPINN.lambda;

if size(eigF,2)~=length(rpmF), eigF=eigF.'; end
if size(eigP,2)~=length(rpmP), eigP=eigP.'; end

freqF=zeros(length(rpmF),2);
freqP=zeros(length(rpmP),2);
zetaF=zeros(length(rpmF),2);
zetaP=zeros(length(rpmP),2);

for k=1:length(rpmF)
    lam=eigF(:,k);
    [~,order]=sort(abs(imag(lam)));
    lam1=lam(order(1)); lam2=lam(order(3));
    freqF(k,:)=[abs(imag(lam1)) abs(imag(lam2))]/(2*pi);
    zetaF(k,:)=[-real(lam1)/max(abs(lam1),1e-12) -real(lam2)/max(abs(lam2),1e-12)];
end

for k=1:length(rpmP)
    lam=eigP(:,k);
    [~,order]=sort(abs(imag(lam)));
    lam1=lam(order(1)); lam2=lam(order(3));
    freqP(k,:)=[abs(imag(lam1)) abs(imag(lam2))]/(2*pi);
    zetaP(k,:)=[-real(lam1)/max(abs(lam1),1e-12) -real(lam2)/max(abs(lam2),1e-12)];
end


%% STABILITY MAP

figure('Position',[120 100 900 600])
hold on

h1=plot(rpmF,zetaF(:,1),'LineWidth',1.6);
h2=plot(rpmF,zetaF(:,2),'LineWidth',1.6);
h3=plot(rpmP,zetaP(:,1),'--','LineWidth',1.6);
h4=plot(rpmP,zetaP(:,2),'--','LineWidth',1.6);
h5=yline(0,'k--','LineWidth',1.3);

xlabel('Rotor Speed [rpm]')
ylabel('Damping Ratio, \zeta')
title('Stability Map')

legend([h1 h2 h3 h4 h5],{'FVM Mode 1','FVM Mode 2','PINN Mode 1','PINN Mode 2','Stability Boundary'},'Location','best')

grid on
box on


%% CAMPBELL DIAGRAM

figure('Position',[120 100 900 600])
hold on

h1=plot(rpmF,freqF(:,1),'LineWidth',1.6);
h2=plot(rpmF,freqF(:,2),'LineWidth',1.6);
h3=plot(rpmP,freqP(:,1),'--','LineWidth',1.6);
h4=plot(rpmP,freqP(:,2),'--','LineWidth',1.6);
h5=plot(rpmF,rpmF/60,'k--','LineWidth',1.5);

xlabel('Rotor Speed [rpm]')
ylabel('Damped Natural Frequency [Hz]')
title('Campbell Diagram')

legend([h1 h2 h3 h4 h5],{'FVM Mode 1','FVM Mode 2','PINN Mode 1','PINN Mode 2','1X'},'Location','northwest')

grid on
box on


%% WATERFALL SETTINGS

omega0=100;
alpha=60;
windowTime=0.50;
overlap=0.75;
skip=2;
fMax=220;


%% FVM WATERFALL

TF=TF(:);
yF=yF(:);

NwF=round(windowTime/mean(diff(TF)));
hopF=round((1-overlap)*NwF);
startF=1:hopF:(length(TF)-NwF+1);

for k=1:length(startF)
    ii=startF(k):startF(k)+NwF-1;
    tseg=TF(ii);
    yseg=yF(ii)-mean(yF(ii));
    [FFT,f]=fftscale(yseg.',tseg.');
    nHalf=floor(length(f)/2)+1;

    if k==1
        fWF=f(1:nHalf);
        ampWF=zeros(length(startF),nHalf);
        tWF=zeros(1,length(startF));
    end

    ampWF(k,:)=abs(FFT(1:nHalf));
    tWF(k)=mean(tseg);
end

rpmWF=(omega0+alpha*tWF)*60/(2*pi);


%% PINN WATERFALL

TP=TP(:);
yP=yP(:);

NwP=round(windowTime/mean(diff(TP)));
hopP=round((1-overlap)*NwP);
startP=1:hopP:(length(TP)-NwP+1);

for k=1:length(startP)
    ii=startP(k):startP(k)+NwP-1;
    tseg=TP(ii);
    yseg=yP(ii)-mean(yP(ii));
    [FFT,f]=fftscale(yseg.',tseg.');
    nHalf=floor(length(f)/2)+1;

    if k==1
        fWP=f(1:nHalf);
        ampWP=zeros(length(startP),nHalf);
        tWP=zeros(1,length(startP));
    end

    ampWP(k,:)=abs(FFT(1:nHalf));
    tWP(k)=mean(tseg);
end

rpmWP=(omega0+alpha*tWP)*60/(2*pi);


%% WATERFALL PLOT

freqMaskF=fWF<=fMax;
freqMaskP=fWP<=fMax;

waterIdxF=1:skip:length(rpmWF);
waterIdxP=1:skip:length(rpmWP);

rpmFplot=rpmWF(waterIdxF);
rpmPplot=rpmWP(waterIdxP);

ZF=ampWF(waterIdxF,freqMaskF);
ZP=ampWP(waterIdxP,freqMaskP);

figure('Position',[50 100 1500 600])

subplot(1,2,1)
waterfall(fWF(freqMaskF),rpmFplot,ZF)
xlabel('Frequency [Hz]')
ylabel('Rotor Speed [rpm]')
zlabel('Amplitude, y/c')
title('FVM Vertical Waterfall')
xlim([0 fMax]); zlim([0 1]); zticks(0:0.1:1)
view(-45,40)
grid on
box on

subplot(1,2,2)
waterfall(fWP(freqMaskP),rpmPplot,ZP)
xlabel('Frequency [Hz]')
ylabel('Rotor Speed [rpm]')
zlabel('Amplitude, y/c')
title('PINN Vertical Waterfall')
xlim([0 fMax]); zlim([0 1]); zticks(0:0.1:1)
view(-45,40)
grid on
box on
%% 
fprintf('\n===== MAX ECCENTRICITY =====\n')
fprintf('FVM  max epsilon = %.6f\n',max(eF))
fprintf('PINN max epsilon = %.6f\n',max(eP))

%% COMPUTATIONAL TIME COMPARISON

timeRotor_hr = [timeFVM timePINN]/3600;
timeStability_hr = [timeStabilityFVM timeStabilityPINN]/3600;

figure('Position',[150 120 1100 450])

subplot(1,2,1)

bar(timeRotor_hr,0.55)

set(gca,'XTickLabel',{'FVM','PINN'},'FontSize',11)

ylabel('Computational Time [hr]')
title('Rotor Response')

text(1,timeRotor_hr(1),sprintf('%.2f hr',timeRotor_hr(1)), ...
    'HorizontalAlignment','center','VerticalAlignment','bottom')

text(2,timeRotor_hr(2),sprintf('%.2f hr',timeRotor_hr(2)), ...
    'HorizontalAlignment','center','VerticalAlignment','bottom')

ylim([0 1.15*max(timeRotor_hr)])

grid on
box on


subplot(1,2,2)

bar(timeStability_hr,0.55)

set(gca,'XTickLabel',{'FVM','PINN'},'FontSize',11)

ylabel('Computational Time [hr]')
title('Stability Analysis')

text(1,timeStability_hr(1),sprintf('%.2f hr',timeStability_hr(1)), ...
    'HorizontalAlignment','center','VerticalAlignment','bottom')

text(2,timeStability_hr(2),sprintf('%.2f hr',timeStability_hr(2)), ...
    'HorizontalAlignment','center','VerticalAlignment','bottom')

ylim([0 1.15*max(timeStability_hr)])

grid on
box on
%% PINN TRAINING HISTORY

load('pinn_model_history.mat','history')

trainingTime_hr = history.elapsedMin/60;

figure('Position',[150 100 900 700])

subplot(2,1,1)

semilogy(history.epoch,history.total)
hold on
semilogy(history.epoch,history.pde)
semilogy(history.epoch,history.bc)
semilogy(history.epoch,history.gal)
semilogy(history.epoch,history.full)

xlabel('Epoch')
ylabel('Loss')

legend('Total','PDE','BC','Galerkin','Full Residual', ...
    'Location','best')

grid on
box on


subplot(2,1,2)

plot(history.epoch,trainingTime_hr,'LineWidth',1.5)

xlabel('Epoch')
ylabel('Elapsed Training Time [hr]')

grid on
box on
%% LOCAL FUNCTIONS

function F=forceFromPressure(q,omega,model)
xbar=linspace(0,1,size(q,2));
zbar=linspace(0,1,size(q,1));
theta=2*pi*xbar;
p=model.mu*omega*(model.R/model.c)^2*q;
Fx=2*pi*model.R*model.L*trapz(zbar,trapz(xbar,p.*sin(theta),2));
Fy=2*pi*model.R*model.L*trapz(zbar,trapz(xbar,p.*cos(theta),2));
F=[Fx;Fy];
end

function u=simpleLHS(n,d)
u=zeros(n,d);
for j=1:d
    p=randperm(n).';
    u(:,j)=(p-1+rand(n,1))/n;
end
end

