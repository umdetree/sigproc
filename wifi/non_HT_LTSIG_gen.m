nSig = 10;
sigLen = 512;

mcss = randi([0, 7], 1, nSig);
res = zeros(sigLen, nSig);
for i = 1:nSig
    cfg = wlanNonHTConfig("MCS", mcss(i));
    psdulength = cfg.PSDULength;
    psdu = randi([0 1], 8 * psdulength, 1);
    waveform = wlanWaveformGenerator(psdu, cfg, "OversamplingFactor", 1);
    res(:, i) = waveform(1:512);
end

save('./wifi-non-HT.mat', "res");