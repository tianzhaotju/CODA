# -*- coding: utf-8 -*-


def tokens2stmts(code, language):
    stmts = [x.strip() for x in code.split('\n') if x.strip()!='']
    paren_n = 0
    stmtInsPos = []
    structStack = []
    if language == 'c':
        for i, stmt in enumerate(stmts):
            if stmt == '{':
                continue
            if stmt.rstrip().endswith("{"):
                paren_n += 1
                if stmt.lstrip().startswith("struct") or stmt.lstrip().startswith("union"):
                    structStack.append((stmt, paren_n))
            elif stmt.lstrip().startswith("}"):
                if structStack != [] and paren_n == structStack[-1][1]:
                    structStack.pop()
                paren_n -= 1
            if structStack == []:
                stmtInsPos.append(i)
    elif language == 'python':
        for i, stmt in enumerate(stmts):
            if stmt.strip()[-1] == '\\':
                continue
            if i == 0 or i == len(stmts)-1:
                continue
            stmtInsPos.append(i)
    elif language == 'java':
        for i, stmt in enumerate(stmts):
            if stmt == '{':
                continue
            if stmt.rstrip().endswith("{"):
                paren_n += 1
                if stmt.lstrip().startswith("class") or stmt.lstrip().startswith("private class") or stmt.lstrip().startswith("public class"):
                    structStack.append((stmt, paren_n))
            elif stmt.lstrip().startswith("}"):
                if structStack != [] and paren_n == structStack[-1][1]:
                    structStack.pop()
                paren_n -= 1
            if structStack == []:
                stmtInsPos.append(i)
    return stmtInsPos


def StmtInsPos(code, stmts_t, language, strict=True):
    statements = stmts_t
    stmtInsPos = tokens2stmts(code, language)
    res = []
    if language == 'c':
        cnt, indent = 0, 0
        if not strict:
            res.append(-1)
        for i, stmt in enumerate(statements):
            cnt += len(stmt)
            if stmt[-1] == "}":
                indent -= 1
            elif stmt[-1] == "{" and stmt[0] not in ["struct", "union", "enum", "typedef"]:
                indent += 1
            if i in stmtInsPos:
                if not strict:
                    res.append(cnt-1)
                elif stmt[-1]!="}" and\
                    indent!=0 and\
                    stmt[0] not in ['else', 'if'] and\
                    not (stmt[0]=='for' and 'if' in stmt):
                    res.append(cnt-1)
    elif language == 'python':
        cnt = 0
        if not strict:
            res.append(-1)
        for i, stmt in enumerate(statements):
            cnt += len(stmt)
            if i in stmtInsPos:
                if not strict:
                    res.append(cnt - 1)
                elif stmt[0] not in ['else', 'if', 'elif'] and \
                        not (stmt[0] == 'for' and 'if' in stmt):
                    res.append(cnt - 1)
    elif language == 'java':
        cnt, indent = 0, 0
        if not strict:
            res.append(-1)
        for i, stmt in enumerate(statements):
            cnt += len(stmt)
            if stmt[-1] == "}":
                indent -= 1
            elif stmt[-1] == "{" and stmt[0] not in ["class"]:
                indent += 1
            if i in stmtInsPos:
                if not strict:
                    res.append(cnt - 1)
                elif stmt[-1] != "}" and \
                        indent != 0 and \
                        stmt[0] not in ['else', 'if'] and \
                        not (stmt[0] == 'for' and 'if' in stmt):
                    res.append(cnt - 1)
    return res


def InsAddCandidates(insertDict, maxLen=None):
    res = []
    for pos in insertDict.keys():
        if pos == "count":
            continue
        if maxLen is None:
            res.append(pos)
        elif int(pos) < maxLen:
            res.append(pos)
    return res


def InsAdd(insertDict, pos, insertedTokenList):
    suc = True
    assert insertDict.get(pos) is not None
    if insertedTokenList in insertDict[pos]:
        suc = False
    else:
        insertDict[pos].append(insertedTokenList)
    if suc:
        if insertDict.get("count") is not None:
            insertDict["count"] += 1
        else:
            insertDict["count"] = 1
    return suc


def InsDeleteCandidates(insertDict):
    res = []
    for key in insertDict.keys():
        if key == "count":
            continue
        if insertDict[key] != []:
            for i, _ in enumerate(insertDict[key]):
                res.append((key, i))
    return res


def InsDelete(insertDict, pos, listIdx=0):
    assert insertDict.get(pos) is not None
    assert insertDict[pos] != []
    if len(insertDict[pos]) <= listIdx:
        assert False
    del insertDict[pos][listIdx]
    assert insertDict.get("count") is not None
    insertDict["count"] -= 1


def InsResult(tokens, insertDict):
    result = []
    if insertDict.get(-1) is not None:
        for tokenList in insertDict[-1]:
            result += tokenList
    for i, t in enumerate(tokens):
        result.append(t)
        if insertDict.get(i) is not None:
            for tokenList in insertDict[i]:
                result += tokenList
    return result