# lambdaEncoder Class

## Overview
The `lambdaEncoder` class provides methods to encode strings into lists of integers based on a specified character set, and to decode lists of integers back into strings using the same character set.

## Usage
### 1. Encoding
#### Method Signature:
```csharp
public List<int> Encode(string Data, List<char> chars)
```
#### Description:
Encodes a given string (`Data`) using a specified list of characters (`chars`). Each character in the string is converted to its corresponding index in the `chars` list.

#### Example:
```csharp
using AroopaApi.Encoders;

lambdaEncoder encoder = new lambdaEncoder();
string inputString = "hello";
List<char> characterSet = new List<char> { 'h', 'e', 'l', 'o' };
List<int> encodedData = encoder.Encode(inputString, characterSet);
Console.WriteLine(string.Join(", ", encodedData));
// encodedData should now contain: [0, 1, 2, 2, 3]
```

### 2. Decoding
#### Method Signature:
```csharp
public string Decode(List<int> Data, List<char> chars)
```
#### Description:
Decodes a given list of integers (`Data`) using a specified list of characters (`chars`). Each integer in the list is converted back to its corresponding character in the `chars` list, forming the original string.
#### Example:
```csharp
using AroopaApi.Encoders;

lambdaEncoder decoder = new lambdaEncoder();
List<int> encodedData1 = new List<int> { 0, 1, 2, 2, 3 };
List<char> characterSet1 = new List<char> { 'h', 'e', 'l', 'o' };
string decodedString = decoder.Decode(encodedData1, characterSet1);
Console.WriteLine(decodedString);
// decodedString should now contain: "hello"
```

```
  +-----------------+
  |   Encode()      |
  | (Input String)  |
  +-----------------+
           |
           v
  +-----------------+
  | Character Set   |
  | (List of Chars) |
  +-----------------+
           |
           v
  +-----------------+
  |  Encode Function|
  |  (Converts String|
  |  to List of Ints)|
  +-----------------+
           |
           v
  +-----------------+
  | Encoded Data    |
  | (List of Ints)  |
  +-----------------+

  (Decoding is the reverse process)

```
