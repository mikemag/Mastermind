// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

//**********************************************************************
// DO NOT EDIT!!
// This file generated by the Python program in scripts.
//**********************************************************************

template <uint8_t p, uint8_t c>
constexpr uint32_t presetInitialGuessKnuth() {
  switch ((p << 4u) | c) {
    case 0x22:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2a:
    case 0x2b:
    case 0x2c:
    case 0x2d:
    case 0x2e:
    case 0x2f:
      return 0x12;
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x35:
      return 0x112;
    case 0x36:
    case 0x37:
    case 0x38:
    case 0x39:
    case 0x3a:
    case 0x3b:
    case 0x3c:
    case 0x3d:
    case 0x3e:
    case 0x3f:
      return 0x123;
    case 0x42:
      return 0x1112;
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x4f:
      return 0x1123;
    case 0x46:
      return 0x1122;
    case 0x47:
    case 0x48:
    case 0x49:
    case 0x4a:
    case 0x4b:
    case 0x4c:
    case 0x4d:
    case 0x4e:
      return 0x1234;
    case 0x52:
    case 0x53:
      return 0x11122;
    case 0x54:
    case 0x55:
    case 0x56:
    case 0x57:
    case 0x58:
      return 0x11223;
    case 0x59:
    case 0x5f:
      return 0x11234;
    case 0x5a:
    case 0x5b:
    case 0x5c:
    case 0x5d:
    case 0x5e:
      return 0x12345;
    case 0x62:
      return 0x111222;
    case 0x63:
      return 0x111223;
    case 0x64:
    case 0x65:
      return 0x112233;
    case 0x66:
      return 0x111234;
    case 0x67:
    case 0x68:
    case 0x69:
    case 0x6b:
      return 0x112234;
    case 0x6a:
    case 0x6c:
    case 0x6f:
      return 0x112345;
    case 0x6d:
    case 0x6e:
      return 0x123456;
    case 0x72:
      return 0x1111222;
    case 0x73:
    case 0x74:
      return 0x1112233;
    case 0x75:
    case 0x79:
      return 0x1112234;
    case 0x76:
    case 0x77:
    case 0x7b:
      return 0x1122334;
    case 0x78:
    case 0x7a:
    case 0x7c:
      return 0x1122345;
    case 0x7d:
      return 0x1123456;
    case 0x82:
      return 0x11111222;
    case 0x83:
      return 0x11122233;
    case 0x84:
    case 0x85:
    case 0x86:
      return 0x11122334;
    case 0x87:
    case 0x88:
      return 0x11223344;
    case 0x89:
      return 0x11223345;
    default:
      return (Codeword<p, c>::ONE_PINS >> p / 2 * 4) + Codeword<p, c>::ONE_PINS;
  }
}
template <uint8_t p, uint8_t c>
constexpr uint32_t presetInitialGuessMostParts() {
  switch ((p << 4u) | c) {
    case 0x22:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2a:
    case 0x2b:
    case 0x2c:
    case 0x2d:
    case 0x2e:
    case 0x2f:
      return 0x12;
    case 0x32:
    case 0x33:
    case 0x34:
      return 0x112;
    case 0x35:
    case 0x36:
    case 0x37:
    case 0x38:
    case 0x39:
    case 0x3a:
    case 0x3b:
    case 0x3c:
    case 0x3d:
    case 0x3e:
    case 0x3f:
      return 0x123;
    case 0x42:
      return 0x1112;
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x47:
      return 0x1123;
    case 0x48:
    case 0x49:
    case 0x4a:
    case 0x4c:
    case 0x4d:
    case 0x4e:
    case 0x4f:
      return 0x1234;
    case 0x4b:
      return 0x1122;
    case 0x52:
    case 0x56:
      return 0x11122;
    case 0x53:
      return 0x11123;
    case 0x54:
    case 0x55:
    case 0x57:
    case 0x58:
      return 0x11223;
    case 0x59:
    case 0x5a:
    case 0x5b:
    case 0x5c:
      return 0x11234;
    case 0x5d:
    case 0x5e:
    case 0x5f:
      return 0x12345;
    case 0x62:
      return 0x111122;
    case 0x63:
    case 0x64:
    case 0x65:
      return 0x111223;
    case 0x66:
    case 0x68:
    case 0x6d:
      return 0x112233;
    case 0x67:
    case 0x69:
    case 0x6a:
    case 0x6b:
    case 0x6c:
      return 0x112234;
    case 0x6e:
      return 0x112345;
    case 0x6f:
      return 0x111234;
    case 0x72:
      return 0x1111222;
    case 0x73:
      return 0x1111223;
    case 0x74:
      return 0x1112223;
    case 0x75:
    case 0x76:
      return 0x1112233;
    case 0x77:
      return 0x1234567;
    case 0x78:
    case 0x79:
    case 0x7a:
    case 0x7b:
    case 0x7d:
      return 0x1122334;
    case 0x7c:
      return 0x1122345;
    case 0x82:
      return 0x11112222;
    case 0x83:
      return 0x11112223;
    case 0x84:
    case 0x85:
    case 0x86:
      return 0x11122233;
    case 0x87:
    case 0x88:
    case 0x89:
      return 0x11122334;
    default:
      return (Codeword<p, c>::ONE_PINS >> p / 2 * 4) + Codeword<p, c>::ONE_PINS;
  }
}
template <uint8_t p, uint8_t c>
constexpr uint32_t presetInitialGuessExpectedSize() {
  switch ((p << 4u) | c) {
    case 0x22:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2a:
    case 0x2b:
    case 0x2c:
    case 0x2d:
    case 0x2e:
    case 0x2f:
      return 0x12;
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x35:
      return 0x112;
    case 0x36:
    case 0x37:
    case 0x38:
    case 0x39:
    case 0x3a:
    case 0x3b:
    case 0x3c:
    case 0x3d:
    case 0x3e:
    case 0x3f:
      return 0x123;
    case 0x42:
      return 0x1112;
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x4a:
      return 0x1123;
    case 0x46:
      return 0x1122;
    case 0x47:
    case 0x48:
    case 0x49:
    case 0x4b:
    case 0x4c:
    case 0x4d:
    case 0x4e:
    case 0x4f:
      return 0x1234;
    case 0x52:
      return 0x11122;
    case 0x53:
      return 0x11123;
    case 0x54:
    case 0x55:
    case 0x56:
    case 0x57:
    case 0x58:
    case 0x5b:
      return 0x11223;
    case 0x59:
    case 0x5a:
    case 0x5c:
    case 0x5e:
    case 0x5f:
      return 0x11234;
    case 0x5d:
      return 0x12345;
    case 0x62:
      return 0x111122;
    case 0x63:
      return 0x111223;
    case 0x64:
    case 0x65:
      return 0x111234;
    case 0x66:
    case 0x69:
      return 0x112233;
    case 0x67:
    case 0x68:
      return 0x112234;
    case 0x6a:
    case 0x6b:
      return 0x123456;
    case 0x6c:
    case 0x6d:
    case 0x6e:
    case 0x6f:
      return 0x112345;
    case 0x72:
      return 0x1111222;
    case 0x73:
      return 0x1112223;
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x77:
      return 0x1112233;
    case 0x78:
    case 0x79:
    case 0x7a:
    case 0x7b:
      return 0x1234567;
    case 0x82:
      return 0x11111222;
    case 0x83:
    case 0x84:
      return 0x11122233;
    case 0x85:
      return 0x11122334;
    case 0x86:
      return 0x11112223;
    case 0x87:
      return 0x11234567;
    case 0x88:
      return 0x12345678;
    default:
      return (Codeword<p, c>::ONE_PINS >> p / 2 * 4) + Codeword<p, c>::ONE_PINS;
  }
}
template <uint8_t p, uint8_t c>
constexpr uint32_t presetInitialGuessEntropy() {
  switch ((p << 4u) | c) {
    case 0x22:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2a:
    case 0x2b:
    case 0x2c:
    case 0x2d:
    case 0x2e:
    case 0x2f:
      return 0x12;
    case 0x32:
    case 0x33:
    case 0x34:
      return 0x112;
    case 0x35:
    case 0x36:
    case 0x37:
    case 0x38:
    case 0x39:
    case 0x3a:
    case 0x3b:
    case 0x3c:
    case 0x3d:
    case 0x3e:
    case 0x3f:
      return 0x123;
    case 0x42:
      return 0x1112;
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x47:
      return 0x1123;
    case 0x48:
    case 0x49:
    case 0x4a:
    case 0x4b:
    case 0x4c:
    case 0x4d:
    case 0x4e:
    case 0x4f:
      return 0x1234;
    case 0x52:
    case 0x56:
      return 0x11122;
    case 0x53:
      return 0x11123;
    case 0x54:
    case 0x55:
    case 0x57:
    case 0x58:
    case 0x5f:
      return 0x11223;
    case 0x59:
    case 0x5a:
    case 0x5b:
    case 0x5c:
      return 0x11234;
    case 0x5d:
    case 0x5e:
      return 0x12345;
    case 0x62:
      return 0x111122;
    case 0x63:
    case 0x64:
    case 0x66:
      return 0x111223;
    case 0x65:
    case 0x67:
    case 0x69:
    case 0x6a:
    case 0x6b:
    case 0x6c:
      return 0x112234;
    case 0x68:
      return 0x112233;
    case 0x6d:
    case 0x6e:
    case 0x6f:
      return 0x112345;
    case 0x72:
      return 0x1111222;
    case 0x73:
      return 0x1112223;
    case 0x74:
    case 0x75:
    case 0x76:
      return 0x1112233;
    case 0x77:
    case 0x78:
    case 0x79:
    case 0x7a:
      return 0x1122334;
    case 0x7b:
      return 0x1122345;
    case 0x82:
      return 0x11112222;
    case 0x83:
    case 0x84:
      return 0x11122233;
    case 0x85:
    case 0x86:
    case 0x87:
      return 0x11122334;
    case 0x88:
      return 0x12345678;
    default:
      return (Codeword<p, c>::ONE_PINS >> p / 2 * 4) + Codeword<p, c>::ONE_PINS;
  }
}
