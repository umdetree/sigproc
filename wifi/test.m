raw_bits = [0;1;0;0;0;1;1;1;0;1;0;1;1;0;0;0;0;0;0;0;1;1;1;0;0;0;0;1;1;0;0;0;0;0;0;0;0;0;1;1;1;0;0;1;1;1;1;1];
deinterleaved_bits = wlanBCCDeinterleave(raw_bits, 'Non-HT', 48);
% wlanBCCDecode(deinterleaved_bits, "1/2");
input_bits = [1;0;1;1;0;0;0;1;0;0;1;1;0;0;0;0;0;0;0;0;0;0;0;0];
encoded_bits = wlanBCCEncode(input_bits, "1/2");
padded_bits = [encoded_bits; zeros(12, 1)]
wlanBCCDecode(padded_bits, "1/2", 'hard')