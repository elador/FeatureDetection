clear
close all;
F = csvread('C:\Users\Patrik\Documents\GitHub\build\match-video\out\04327d2442.txt');
numPosPairs = F(2, 1);
numNegPairs = F(2, 2);
numFrames = 1:size(F, 1) - 1; % -1 because of the header line

figure(1);
subplot(1,2,1);
hold on;

for p = 3:numPosPairs+2 % plot all positive pairs
    plot(numFrames, F(2:end, p), 'g', 'DisplayName', sprintf('%d', p-3)); % 2:end because we skip the header line
end
for p = (numPosPairs+2+1):(size(F, 2)-1) % plot all negative pairs
    plot(numFrames, F(2:end, p), 'r', 'DisplayName', sprintf('%d', p-3)); % 2:end because we skip the header line
end
legend('-DynamicLegend');

disp 'Loading video';
videoPlayer = vision.VideoPlayer;
videoFReader = vision.VideoFileReader('Z:\datasets\multiview02\PaSC\training\video\04327d2442.mp4');
for i = 1:numFrames(end)
    frame(i).data = step(videoFReader);
end
release(videoFReader);
release(videoPlayer);
disp 'Finished loading video';

for nn = 1:100
    x = ginput(1);
    clickedFrameNumber = round(x(1));
    subplot(1,2,2);
    imshow(frame(clickedFrameNumber).data);
end

