import argparse
import random
import sys
from keyword import iskeyword
from parser_folder.DFG_python import DFG_python
from parser_folder.DFG_c import DFG_c
from parser_folder.DFG_java import DFG_java
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,)
from tree_sitter import Language, Parser
import javalang
sys.path.append('..')
sys.path.append('../../../')
sys.path.append('.')
sys.path.append('../')


python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">",
                   '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
                   'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global',
                   'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
                   'while', 'with', 'yield', 'int', 'print', 'map', 'range', 'len', 'read_line', 'raw_input', 'xrange',
                   'zip', 'deque', 'float', ]
java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while", 'throw', 'null', 'false', 'true', 'println', 'float']
java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Long", "Float", "Double", "Character",
                    "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object", "String", "StringBuffer",
                    "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet", "HashMap",
                    'Throwable', 'Class', 'InputStream', 'OutputStream', 'URL', 'HttpServletRequest', 'PrintWriter', 'HttpServletResponse',
                    'Thread', 'File', 'IOUtils', 'DataType', 'Writer', 'URLConnection', 'FileOutputStream', 'Browser', 'Connection',
                    'Arrays', 'Reader', 'HttpURLConnection']
c_keywords = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__", "uint8_t", "uint16_t",
                "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t",
                "int64_t", "bool"]
c_macros = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",
              "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
              "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]
c_special_ids = ["main",
                   "stdio", "cstdio", "stdio.h",
                   "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",
                   "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush",
                   "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                   "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                   "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                   "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                   "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                   "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                   "stdlib", "cstdlib", "stdlib.h",
                   "size_t", "div_t", "ldiv_t", "lldiv_t",
                   "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",
                   "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                   "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                   "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                   "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                   "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                   "mbstowcs", "wcstombs",
                   "string", "cstring", "string.h",
                   "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",
                   "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                   "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                   "strpbrk" ,"strstr", "strtok", "strxfrm",
                   "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",
                   "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                   "iostream", "istream", "ostream", "fstream", "sstream",
                   "iomanip", "iosfwd",
                   "ios", "wios", "streamoff", "streampos", "wstreampos",
                   "streamsize", "cout", "cerr", "clog", "cin",
                   "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",
                   "noshowbase", "showpoint", "noshowpoint", "showpos",
                   "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                   "left", "right", "internal", "dec", "oct", "hex", "fixed",
                   "scientific", "hexfloat", "defaultfloat", "width", "fill",
                   "precision", "endl", "ends", "flush", "ws", "showpoint",
                   "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",
                   "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                   "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                   "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]
special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/',
                '|']


def is_valid_variable_python(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif iskeyword(name):
        return False
    elif name in python_keywords:
        return False
    return True


def is_valid_variable_java(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in java_keywords:
        return False
    elif name in java_special_ids:
        return False
    return True


def is_valid_variable_c(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in c_keywords:
        return False
    elif name in c_macros:
        return False
    elif name in c_special_ids:
        return False
    return True


def is_valid_variable_name(name: str, lang: str) -> bool:
    if lang == 'python':
        return is_valid_variable_python(name)
    elif lang == 'c':
        return is_valid_variable_c(name)
    elif lang == 'java':
        return is_valid_variable_java(name)
    else:
        return False

path = '../../../python_parser/parser_folder/my-languages.so'
c_code = """static int bit8x8_c(MpegEncContext *s, uint8_t *src1, uint8_t *src2,\n\n                    ptrdiff_t stride, int h)\n\n{\n\n    const uint8_t *scantable = s->intra_scantable.permutated;\n\n    LOCAL_ALIGNED_16(int16_t, temp, [64]);\n\n    int i, last, run, bits, level, start_i;\n\n    const int esc_length = s->ac_esc_length;\n\n    uint8_t *length, *last_length;\n\n\n\n    av_assert2(h == 8);\n\n\n\n    s->pdsp.diff_pixels(temp, src1, src2, stride);\n\n\n\n    s->block_last_index[0 /* FIXME */] =\n\n    last                               =\n\n        s->fast_dct_quantize(s, temp, 0 /* FIXME */, s->qscale, &i);\n\n\n\n    bits = 0;\n\n\n\n    if (s->mb_intra) {\n\n        start_i     = 1;\n\n        length      = s->intra_ac_vlc_length;\n\n        last_length = s->intra_ac_vlc_last_length;\n\n        bits       += s->luma_dc_vlc_length[temp[0] + 256]; // FIXME: chroma\n\n    } else {\n\n        start_i     = 0;\n\n        length      = s->inter_ac_vlc_length;\n\n        last_length = s->inter_ac_vlc_last_length;\n\n    }\n\n\n\n    if (last >= start_i) {\n\n        run = 0;\n\n        for (i = start_i; i < last; i++) {\n\n            int j = scantable[i];\n\n            level = temp[j];\n\n\n\n            if (level) {\n\n                level += 64;\n\n                if ((level & (~127)) == 0)\n\n                    bits += length[UNI_AC_ENC_INDEX(run, level)];\n\n                else\n\n                    bits += esc_length;\n\n                run = 0;\n\n            } else\n\n                run++;\n\n        }\n\n        i = scantable[last];\n\n\n\n        level = temp[i] + 64;\n\n\n\n        av_assert2(level - 64);\n\n\n\n        if ((level & (~127)) == 0)\n\n            bits += last_length[UNI_AC_ENC_INDEX(run, level)];\n\n        else\n\n            bits += esc_length;\n\n    }\n\n\n\n    return bits;\n\n}\n"""
python_code = """def solve():\n      h, w, m = map(int, raw_input().split())\n      if h == 1:\n          print 'c' + '.' * (h * w - m - 1) + '*' * m\n      elif w == 1:\n          for c in 'c' + '.' * (h * w - m - 1) + '*' * m:\n              print c\n      elif h * w - m == 1:\n          print 'c' + '*' * (w - 1)\n          for _ in xrange(h-1):\n              print '*' * w\n      else:\n          m = h * w - m\n          for i in xrange(h-1):\n              for j in xrange(w-1):\n                  t = (i + 2) * 2 + (j + 2) * 2 - 4\n                  r = (i + 2) * (j + 2)\n                  if t <= m <= r:\n                      a = [['*'] * w for _ in xrange(h)]\n                      for k in xrange(i+2):\n                          a[k][0] = '.'\n                          a[k][1] = '.'\n                      for k in xrange(j+2):\n                          a[0][k] = '.'\n                          a[1][k] = '.'\n                      for y, x in product(range(2, i+2), range(2, j+2)):\n                          if y == 1 and x == 1:\n                              continue\n                          if t >= m:\n                              break\n                          a[y][x] = '.'\n                          t += 1\n                      a[0][0] = 'c'\n                      for s in a:\n                          print ''.join(s)\n                      return\n          print 'Impossible'\n  for t in xrange(int(raw_input())):\n      print "Case #%d:" % (t + 1)\n      solve()\n"""
java_code = """public static void copyFile(File in, File out) throws IOException {\n        FileChannel inChannel = new FileInputStream(in).getChannel();\n        FileChannel outChannel = new FileOutputStream(out).getChannel();\n        try {\n            inChannel.transferTo(0, inChannel.size(), outChannel);\n        } catch (IOException e) {\n            throw e;\n        } finally {\n            if (inChannel != null) inChannel.close();\n            if (outChannel != null) outChannel.close();\n        }\n    }\n"""
dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'c': DFG_c,
}

parsers = {}
for lang in dfg_function:
    LANGUAGE = Language(path, lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser
codes = {
    'python': python_code,
    'java': java_code,
    'c': c_code,
}


def get_code_tokens(code, lang):
    code = code.split('\n')
    code_tokens = [x + '\\n' for x in code if x ]
    return code_tokens


def extract_dataflow(code, lang):
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)
    index_table = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_table[idx] = index
    DFG, _ = parser[1](root_node, index_to_code, {})
    DFG = sorted(DFG, key=lambda x: x[1])
    return DFG, index_table, code_tokens


def get_example(code, tgt_word, substitute, lang):
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    replace_pos = {}
    for index, code_token in enumerate(code_tokens):
        if code_token == tgt_word:
            try:
                replace_pos[tokens_index[index][0][0]].append((tokens_index[index][0][1], tokens_index[index][1][1]))
            except:
                replace_pos[tokens_index[index][0][0]] = [(tokens_index[index][0][1], tokens_index[index][1][1])]
    diff = len(substitute) - len(tgt_word)
    for line in replace_pos.keys():
        for index, pos in enumerate(replace_pos[line]):
            code[line] = code[line][:pos[0]+index*diff] + substitute + code[line][pos[1]+index*diff:]
    return "\n".join(code)


def get_example_batch(code, chromesome, lang):
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    replace_pos = {}
    for tgt_word in chromesome.keys():
        diff = len(chromesome[tgt_word]) - len(tgt_word)
        for index, code_token in enumerate(code_tokens):
            if code_token == tgt_word:
                try:
                    replace_pos[tokens_index[index][0][0]].append((tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1]))
                except:
                    replace_pos[tokens_index[index][0][0]] = [(tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1])]
    for line in replace_pos.keys():
        diff = 0
        for index, pos in enumerate(replace_pos[line]):
            code[line] = code[line][:pos[3]+diff] + pos[1] + code[line][pos[4]+diff:]
            diff += pos[2]
    return "\n".join(code)


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def remove_strings(code):
    a = -1
    b = -1
    while(code.find('"', b+1) != -1):
        a = code.find('"', b+1)
        b = code.find('"', a+1)
        if b == -1:
            break
        code = code[:a+1] + code[b:]
    a = -1
    b = -1
    while (code.find("'", b + 1) != -1):
        a = code.find("'", b + 1)
        b = code.find("'", a + 1)
        if b == -1:
            break
        code = code[:a + 1] + code[b:]
    return code


def get_identifiers_ori(code, lang):
    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    ret = set()
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret.add(d[0])
    return list(ret), code_tokens


def get_identifiers(code, lang, only_code_tokens=False):
    code = code.replace("\\n", "\n")
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    try:
        code = remove_strings(code)
    except:
        pass
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code_tokens = [index_to_code_token(x, code.split('\n')) for x in tokens_index]
    if only_code_tokens:
        return code_tokens
    ret = []
    if lang == 'python':
        dfg, index_table, _ = extract_dataflow(code, lang)
        variable_name = []
        function_name = []
        for d in dfg:
            if is_valid_variable_name(d[0], lang):
                variable_name.append(d[0])
        for c_i, c in enumerate(code_tokens):
            if c_i - 1 >= 0 and code_tokens[c_i-1] == 'def':
                function_name.append(c)
        return list(set(variable_name)), list(set(function_name)), code_tokens
    elif lang == 'java':
        dfg, _, _ = extract_dataflow(code, lang)
        variables = set()
        for d in dfg:
            if is_valid_variable_name(d[0], lang):
                variables.add(d[0])
        variables = list(variables)
        variable_name = []
        function_name = []
        for var in variables:
            name = 'variable_name'
            for i in range(len(code_tokens)):
                if code_tokens[i] == var:
                    if (i < len(code_tokens) - 1 and code_tokens[i + 1] == '('):
                        name = 'function_name'
                        break
                    if i < len(code_tokens) - 2 and code_tokens[i + 1] == '' and code_tokens[i + 2] == '(':
                        name = 'function_name'
                        break
            if name == 'variable_name':
                variable_name.append(var)
            elif name == 'function_name':
                function_name.append(var)
        return variable_name, function_name, code_tokens
    elif lang == 'c':
        variables = set()
        for c in code_tokens:
            if is_valid_variable_name(c, lang):
                variables.add(c)
        variables = list(variables)
        variable_name = []
        function_name = []
        for var in variables:
            name = 'variable_name'
            for i in range(len(code_tokens)):
                if code_tokens[i] == var:
                    if (i < len(code_tokens) - 1 and code_tokens[i + 1] == '('):
                        name = 'function_name'
                        break
                    if i < len(code_tokens) - 2 and code_tokens[i + 1] == '' and code_tokens[i + 2] == '(':
                        name = 'function_name'
                        break
            if name == 'variable_name':
                variable_name.append(var)
            elif name == 'function_name':
                function_name.append(var)
        return variable_name, function_name, code_tokens


def get_code_style(subs_code, lang):
    code_style = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    if lang == 'c':
        for target_code in subs_code:
            for c_print in ['printf(']:
                a = target_code.count(c_print+'"')
                code_style[0][0] += a
                code_style[0][1] += target_code.count(c_print) - a
            for combined_assignment in ['+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=']:
                code_style[1][0] += target_code.count(combined_assignment[:-1])
                code_style[1][1] += target_code.count(combined_assignment)
            for unary_operator in ['++', '--']:
                a = target_code.count(unary_operator)
                code_style[2][0] += target_code.count(unary_operator[0]) - a
                code_style[2][1] += a
            code_style[4][0] += target_code.count('while')
            code_style[4][1] += target_code.count('for')
    elif lang == 'java':
        for target_code in subs_code:
            for java_print in ['System.out.println(', 'System.out.print(']:
                a = target_code.count(java_print+'"')
                code_style[0][0] += a
                code_style[0][1] += target_code.count(java_print) - a
            for combined_assignment in ['+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=']:
                code_style[1][0] += target_code.count(combined_assignment[:-1])
                code_style[1][1] += target_code.count(combined_assignment)
            for unary_operator in ['++', '--']:
                a = target_code.count(unary_operator)
                code_style[2][0] += target_code.count(unary_operator[0]) - a
                code_style[2][1] += a
            code_style[4][0] += target_code.count('while')
            code_style[4][1] += target_code.count('for')
    elif lang == 'python':
        for target_code in subs_code:
            for python_print in ['print ', 'print(']:
                a = target_code.count(python_print+'"')
                a += target_code.count(python_print + "'")
                code_style[0][0] += a
                code_style[0][1] += target_code.count(python_print) - a
            for combined_assignment in ['+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=']:
                code_style[1][0] += target_code.count(combined_assignment[:-1])
                code_style[1][1] += target_code.count(combined_assignment)
            code_style[4][0] += target_code.count('while')
            code_style[4][1] += target_code.count('for')
    for i in range(len(code_style)):
        if code_style[i][0] == 0 and code_style[i][1] == 0:
            code_style[i][0] = 0.5
            code_style[i][1] = 0.5
        else:
            code_style[i][0] = code_style[i][0] / (code_style[i][0]+code_style[i][1])
            code_style[i][1] = code_style[i][1] / (code_style[i][0]+code_style[i][1])
    return code_style


def change_code_style(source_code: str, lang: str, all_variable_name: list, code_style):
    all_source_codes = [source_code]
    _, _, source_code_tokens = get_identifiers(source_code, 'c')
    used_identifiers = []
    used_identifiers.extend(source_code_tokens)
    all_variable_name = list(set(all_variable_name) - (set(all_variable_name) & set(used_identifiers)))
    if random.random() <= code_style[4][1]:
        if lang in ['c', 'java']:
            source_code_list = source_code.split('\n')
            new_source_code_list = []
            i = 0
            while i < len(source_code_list):
                line = source_code_list[i]
                if line.find('for') != -1 and line.split('for')[1].strip()[0]=='(':
                    new_source_code_list.append('{ ' + line.split(';')[0].split('(')[-1]+';')
                    new_source_code_list.append('while('+line.split(';')[1]+'){')
                    start = 0
                    if line.strip()[-1] == '{':
                        start = 1
                        new_source_code_list.append('{')
                    j = i + 1
                    while j < len(source_code_list):
                        new_source_code_list.append(source_code_list[j])
                        start += source_code_list[j].count('{')
                        start -= source_code_list[j].count('}')
                        j += 1
                        if start == 0:
                            break
                    i = j
                    new_source_code_list.append(line.split(';')[2].split(')')[0]+';}')
                    new_source_code_list.append('}')
                else:
                    new_source_code_list.append(line)
                    i += 1
    else:
        if lang in ['c', 'java']:
            source_code_list = source_code.split('\n')
            new_source_code_list = []
            i = 0
            while i < len(source_code_list):
                line = source_code_list[i]
                if line.find('while') != -1:
                    new_source_code_list.append('for(;'+line.split('while')[1].split('(')[1].split(')')[0]+';)')
                    if line.strip()[-1] == '{':
                        new_source_code_list.append('{')
                else:
                    new_source_code_list.append(line)
                i += 1
        all_source_codes.append(source_code)
    if random.random() <= code_style[0][0]:
        if lang == 'c':
            begin = 0
            end = 0
            while source_code.find('printf("', end) != -1:
                begin = source_code.find('printf("', end)
                end = source_code.find('")', begin)
                if end == -1:
                    break
                pre_code = source_code[:begin]
                string = source_code[begin+7:end+1]
                after_code = source_code[end+2:]

                source_code = pre_code + 'char ' + all_variable_name[0] + '['+ str(len(string)) +'] = ' + string + ';' + \
                              'printf("%s",' + all_variable_name[0] + ')' + after_code
                if len(all_variable_name) > 1:
                    all_variable_name = all_variable_name[1:]
                else:
                    all_variable_name[0] += '_'
        elif lang == 'java':
            all_java_print = ['System.out.println', 'System.out.print']
            for java_print in all_java_print:
                begin = 0
                end = 0
                while source_code.find(java_print+'("', end) != -1:
                    begin = source_code.find(java_print+'("', end)
                    end = source_code.find('")', begin)
                    if end == -1:
                        break
                    pre_code = source_code[:begin]
                    string = source_code[begin + len(java_print)+1:end+1]
                    after_code = source_code[end + 2:]

                    source_code = pre_code + 'String ' + all_variable_name[0] + ' = ' + string + ';' + \
                                  java_print + '(' + all_variable_name[0] + ')' + after_code
                    if len(all_variable_name) > 1:
                        all_variable_name = all_variable_name[1:]
                    else:
                        all_variable_name[0] += '_'
        elif lang == 'python':
            source_code_list = source_code.split('\n')
            new_source_code_list = []
            for line in source_code_list:
                temp_line = line.strip()
                if len(temp_line) <= 5:
                    new_source_code_list.append(line)
                    continue
                if temp_line[:5] == 'print':
                    blank = line[:line.find('print')]
                    if temp_line[5] == '(':
                        content = temp_line.split('(')[-1].split(')')[0]
                        if content.find(',') != -1:
                            content.replace(',', '+ " " +')

                        new_source_code_list.append(blank + '%s = %s' % (all_variable_name[0], content))
                        new_source_code_list.append(blank + 'print(%s)' % all_variable_name[0])
                        if len(all_variable_name) > 1:
                            all_variable_name = all_variable_name[1:]
                        else:
                            all_variable_name[0] += '_'
                    elif temp_line[5] == ' ':
                        content = temp_line[6:]
                        if content.find(',') != -1:
                            content.replace(',', '+ " " +')
                        new_source_code_list.append(blank + '%s = %s' % (all_variable_name[0], content))
                        new_source_code_list.append(blank + 'print %s' % all_variable_name[0])
                        if len(all_variable_name) > 1:
                            all_variable_name = all_variable_name[1:]
                        else:
                            all_variable_name[0] += '_'
                    else:
                        new_source_code_list.append(line)
                        continue
                else:
                    new_source_code_list.append(line)
            source_code = '\n'.join(new_source_code_list)
        all_source_codes.append(source_code)
    if random.random() <= code_style[1][0]:
        if lang in ['c', 'java', 'python']:
            all_combined_assignments = ['+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=']
            for combined_assignments in all_combined_assignments:
                index = 0
                while source_code.find(combined_assignments, index) != -1:
                    index = source_code.find(combined_assignments, index)
                    if index == -1:
                        break
                    pre_code = source_code[:index]
                    after_code = source_code[index + len(combined_assignments):]
                    temp_var = source_code[:index].strip().split('\n')[-1].split(';')[-1].split('(')[-1].split(' ')[-1]
                    source_code = pre_code + temp_var + '=' + temp_var + combined_assignments[:-1] + after_code
        all_source_codes.append(source_code)
    if random.random() <= code_style[2][0]:
        if lang in ['c', 'java']:
            all_unary_operator = ['++', '--']
            for unary_operator in all_unary_operator:
                index = 0
                while source_code.find(unary_operator, index) != -1:
                    index = source_code.find(unary_operator, index)
                    if index == -1:
                        break
                    if index+2 < len(source_code) and source_code[index+2] == unary_operator[0]:
                        index += 3
                    pre_code = source_code[:index]
                    after_code = source_code[index + len(unary_operator):]
                    temp_var = source_code[:index].strip().strip().split('\n')[-1].split(';')[-1].split('(')[-1].split(' ')[-1]
                    source_code = pre_code + '=' + temp_var + unary_operator[0] + '1' + after_code
        all_source_codes.append(source_code)
    if random.random() <= code_style[3][0]:
        if lang == 'c':
            source_code_list = source_code.split('\n')
            new_source_code_list = []
            for line in source_code_list:
                temp_line = line
                begin = 0
                for flag in ['true', 'false']:
                    while temp_line.find(flag, begin) != -1:
                        begin = temp_line.find(flag, begin)
                        new_source_code_list.append('bool %s = %s;' % (all_variable_name[0], flag))
                        temp_line = temp_line.replace(flag, all_variable_name[0])
                        if len(all_variable_name) > 1:
                            all_variable_name = all_variable_name[1:]
                        else:
                            all_variable_name[0] += '_'
                tokens = javalang.tokenizer.tokenize(temp_line)
                try:
                    for i in list(tokens):
                        value = str(i.value).strip()
                        if value.isdigit():
                            new_source_code_list.append('int %s = %s;' % (all_variable_name[0], value))
                            temp_line = temp_line.replace(value, all_variable_name[0])
                            if len(all_variable_name) > 1:
                                all_variable_name = all_variable_name[1:]
                            else:
                                all_variable_name[0] += '_'

                        if len(value) >= 2 and (value[0] == '"' and value[-1] == '"'):
                            new_source_code_list.append('char %s[%d] = %s;' % (all_variable_name[0], len(value), value))
                            temp_line = temp_line.replace(value, all_variable_name[0])
                            if len(all_variable_name) > 1:
                                all_variable_name = all_variable_name[1:]
                            else:
                                all_variable_name[0] += '_'
                except:
                    pass
                new_source_code_list.append(temp_line)
            source_code = '\n'.join(new_source_code_list)
        elif lang == 'java':
            source_code_list = source_code.split('\n')
            new_source_code_list = []
            for line in source_code_list:
                temp_line = line
                begin = 0
                for flag in ['true', 'false']:
                    while temp_line.find(flag, begin) != -1:
                        begin = temp_line.find(flag, begin)
                        new_source_code_list.append('boolean %s = %s;' % (all_variable_name[0], flag))
                        temp_line = temp_line.replace(flag, all_variable_name[0])
                        if len(all_variable_name) > 1:
                            all_variable_name = all_variable_name[1:]
                        else:
                            all_variable_name[0] += '_'
                tokens = javalang.tokenizer.tokenize(temp_line)
                try:
                    for i in list(tokens):
                        value = str(i.value).strip()
                        if value.isdigit():
                            new_source_code_list.append('int %s = %s;' % (all_variable_name[0], value))
                            temp_line = temp_line.replace(value, all_variable_name[0])
                            if len(all_variable_name) > 1:
                                all_variable_name = all_variable_name[1:]
                            else:
                                all_variable_name[0] += '_'

                        if len(value) >= 2 and (value[0] == '"' and value[-1] == '"'):
                            new_source_code_list.append('String %s = %s;' % (all_variable_name[0], value))
                            temp_line = temp_line.replace(value, all_variable_name[0])
                            if len(all_variable_name) > 1:
                                all_variable_name = all_variable_name[1:]
                            else:
                                all_variable_name[0] += '_'
                except:
                    pass
                new_source_code_list.append(temp_line)
            source_code = '\n'.join(new_source_code_list)
        elif lang == 'python':
            source_code_list = source_code.split('\n')
            new_source_code_list = []
            for line in source_code_list:
                temp_line = line
                begin = 0
                for flag in ['true', 'false']:
                    while temp_line.find(flag, begin) != -1:
                        begin = temp_line.find(flag, begin)
                        new_source_code_list.append('%s = %s;' % (all_variable_name[0], flag))
                        temp_line = temp_line.replace(flag, all_variable_name[0])
                        if len(all_variable_name) > 1:
                            all_variable_name = all_variable_name[1:]
                        else:
                            all_variable_name[0] += '_'
                tokens = javalang.tokenizer.tokenize(temp_line)
                try:
                    for i in list(tokens):
                        value = str(i.value).strip()
                        if value.isdigit():
                            new_source_code_list.append('%s = %s;' % (all_variable_name[0], value))
                            temp_line = temp_line.replace(value, all_variable_name[0])
                            if len(all_variable_name) > 1:
                                all_variable_name = all_variable_name[1:]
                            else:
                                all_variable_name[0] += '_'

                        if len(value) >= 2 and (value[0] == '"' and value[-1] == '"'):
                            new_source_code_list.append('%s = %s;' % (all_variable_name[0], value))
                            temp_line = temp_line.replace(value, all_variable_name[0])
                            if len(all_variable_name) > 1:
                                all_variable_name = all_variable_name[1:]
                            else:
                                all_variable_name[0] += '_'
                except:
                    pass
                new_source_code_list.append(temp_line)
            source_code = '\n'.join(new_source_code_list)
        all_source_codes.append(source_code)
    return all_source_codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")
    args = parser.parse_args()
    code = codes[args.lang]
    data, _ = get_identifiers(code, args.lang)
    code_ = get_example(java_code, "inChannel", "dwad", "java")
    code_ = get_example_batch(java_code, {"inChannel":"dwad", "outChannel":"geg"}, "java")
    print(code_)


if __name__ == '__main__':
    main()