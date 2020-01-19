%% load all of the auditory data
ft_defaults
datadir='S:/Lab-Shared/Experiments/HighDenseGamma/results/EEG/Auditory/Raw/';
file_names = ["aud_s2_1.bdf","aud_s2_2.bdf","aud_s2_3.bdf","aud_s2_4.bdf"];
savedir=[];%'./PP/'; % directory to save results
data_array = cell(size(file_names));
header_array = cell(size(file_names));
event_array = cell(size(file_names));
for i=1:length(file_names)
    header_array{i} = ft_read_header([datadir,convertStringsToChars(file_names(i))]);
    data_array{i} = ft_read_data([datadir,convertStringsToChars(file_names(i))])';
    event_array{i} = ft_read_event([datadir,convertStringsToChars(file_names(i))])';
end

%% save things for easy access
blk = 4;
channel = 'A23';

chan_num = find(strcmp(header_array{blk}.label,channel));
data = data_array{blk}(:,chan_num);

%get indices of onsets
onsets = [];
for i=1:length(event_array{blk})
    if event_array{blk}(i).value==12
        onsets = [onsets;event_array{blk}(i).sample];
    elseif event_array{blk}(i).value==22
        onsets = [onsets;event_array{blk}(i).sample];
    end    
end

%% plot our chosen channel
ERPfigure();subnum=1;nsubplts=4;
subplot(nsubplts,1,subnum)
plot(data)
title('Data')
subnum=subnum+1;

%% do robust detrending - polynomial

ord = 10;
tit = sprintf('Ord %d',ord);
[y,w,r] = nt_detrend(data,ord);

subplot(nsubplts,1,subnum)
plot(y);hold on
scatter(find(~w),ones(1,length(find(~w))),'r*')
title(tit)
subnum=subnum+1;

%% robust detrending no erp
ord = 10;
window = 0:round(0.5*2048); %num of timepoints to exclude 

tit = sprintf('Ord %d, no ERP',ord);
w=ones(size(data));
w(onsets+window)=0;
[y,w,r] = nt_detrend(data,ord,w);

subplot(nsubplts,1,subnum)
plot(y);hold on
scatter(find(~w),ones(1,length(find(~w))),'r*')
title(tit)
subnum=subnum+1;

%% do robust detrending - sinusoids

ord = 6;
[y,w,r]=nt_detrend(data,ord,[],'sinusoids');
tit = sprintf('Ord %d, sinusoids',ord);

subplot(nsubplts,1,subnum)
plot(y);hold on
scatter(find(~w),ones(1,length(find(~w))),'r*')
title(tit)
subnum=subnum+1;

%% compare to HPF
cutoff = 0.1;

y = HPF(data,2048,cutoff);
tit = sprintf('HPF %0.1fHz',cutoff);

subplot(nsubplts,1,subnum)
plot(y);
title(tit)
subnum=subnum+1;