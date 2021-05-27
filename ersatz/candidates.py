import re

# sentence ending punctuation
# U+0964  ।   Po  DEVANAGARI DANDA
# U+061F  ؟   Po  ARABIC QUESTION MARK
# U+002E  .   Po  FULL STOP
# U+3002  。  Po  IDEOGRAPHIC FULL STOP
# U+0021  !   Po  EXCLAMATION MARK
# U+06D4  ۔   Po  ARABIC FULL STOP
# U+17D4  ។   Po  KHMER SIGN KHAN
# U+003F  ?   Po  QUESTION MARK
# U+2026 ...  Po  Ellipsis
# U+30FB 
# U+002A *

# other acceptable punctuation
# U+3011  】  Pe  RIGHT BLACK LENTICULAR BRACKET
# U+00BB  »   Pf  RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
# U+201D  "   Pf  RIGHT DOUBLE QUOTATION MARK
# U+300F  』  Pe  RIGHT WHITE CORNER BRACKET
# U+2018  ‘   Pi  LEFT SINGLE QUOTATION MARK
# U+0022  "   Po  QUOTATION MARK
# U+300D  」  Pe  RIGHT CORNER BRACKET
# U+201C  "   Pi  LEFT DOUBLE QUOTATION MARK
# U+0027  '   Po  APOSTROPHE
# U+2019  ’   Pf  RIGHT SINGLE QUOTATION MARK
# U+0029  )   Pe  RIGHT PARENTHESIS

ending_punc = {
    '\u0964',
    '\u061F',
    '\u002E',
    '\u3002',
    '\u0021',
    '\u06D4',
    '\u17D4',
    '\u003F',
    '\uFF61',
    '\uFF0E',
    '\u2026',
}

closing_punc = {
    '\u3011',
    '\u00BB',
    '\u201D',
    '\u300F',
    '\u2018',
    '\u0022',
    '\u300D',
    '\u201C',
    '\u0027',
    '\u2019',
    '\u0029'
}

list_set = {
    '\u30fb',
    '\uFF65',
    '\u002a', # asterisk
    '\u002d',
    '\u4e00' 
}

class Split():
    def __call__(self, left_context, right_context):
        return True


class PunctuationSpace(Split):
    def __call__(self, left_context, right_context):
        if right_context[0] == ' ':
            regex = '.*[?!.][.?!")\']*'
            regex = re.compile(regex)
            if regex.fullmatch(left_context) is not None:
                return True
        return False

class Lists(Split):
    def __call__(self, left_context, right_context):
        if right_context.strip()[0] in ['*', '-', '~']:
            return True

class MultilingualPunctuation(Split):
    def __call__(self, left_context, right_context):
        try:
            left_context = left_context.split(' ')[-1]
            if right_context[0] not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                for i, ch in enumerate(left_context):
                    if ch in ending_punc:
                        for j, next_ch in enumerate(left_context[i:], i):
                            if next_ch not in ending_punc and next_ch not in closing_punc:
                                j = -1
                                break
                        if j != -1:
                            return True
        except:
            return False
        return False

class IndividualPunctuation(Split):
    def __init__(self, unicode_char):
        self.punc = unicode_char

    def __call__(self, left_context, right_context):
        if right_context[0] == ' ':
            for i, ch in enumerate(left_context):
                if ch == self.punc:
                    for j, next_ch in enumerate(left_context[i:], i):
                        if next_ch != self.punc and next_ch not in closing_punc:
                            j = -1
                            break
                    if j != -1:
                        return True
        return False
