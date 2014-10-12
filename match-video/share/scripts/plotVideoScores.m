clear
close all;
F = csvread('C:\Users\Patrik\Documents\GitHub\build\match-video\out\06351d37.txt');
% Find the num pos and neg pairs:
numPosPairs = 0;
numNegPairs = 0;
for i = 1:size(F, 1);
    if (F(i, 1) ~= 0)
        numPosPairs = F(i, 1);
    end
    if (F(i, 2) ~= 0)
        numNegPairs = F(i, 2);
    end
end
numFrames = 1:size(F, 1);

figure(1);
subplot(1,2,1);
hold on;

imageNames_invalid = F(:,3:2:end-1);  % odd matrix. invalid because csvread doesn't read text
scores = F(:,4:2:end-1);  % even matrix

for p = 1:numPosPairs % plot all positive pairs
    plot(numFrames, scores(:, p), 'g', 'DisplayName', sprintf('%d', p-1));
end
for p = (numPosPairs+1):size(scores, 2) % plot all negative pairs
    plot(numFrames, scores(:, p), 'r', 'DisplayName', sprintf('%d', p-1));
end
legend('-DynamicLegend');

disp 'Loading video';
videoPlayer = vision.VideoPlayer;
videoFReader = vision.VideoFileReader('Z:\datasets\multiview02\PaSC\training\video\06351d37.mp4');
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

