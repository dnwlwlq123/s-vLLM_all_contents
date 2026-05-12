#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dlfcn.h>
#include <libgen.h>
#include <stdlib.h>

static int mkdir_p(const char *path) {
    char tmp[4096]; snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
    }
    return mkdir(tmp, 0755);
}

static void ensure_parent(const char *p) {
    if (!p) return;
    if (strstr(p, "/.tensorrt_llm/cache/") || strstr(p, "/.cache/") ) {
        char buf[4096]; snprintf(buf, sizeof(buf), "%s", p);
        mkdir_p(dirname(buf));
    }
}

typedef int (*rename_fn_t)(const char *, const char *);
int rename(const char *oldpath, const char *newpath) {
    static rename_fn_t real = NULL;
    if (!real) real = (rename_fn_t)dlsym(RTLD_NEXT, "rename");
    ensure_parent(newpath);
    return real(oldpath, newpath);
}

typedef int (*renameat_fn_t)(int, const char *, int, const char *);
int renameat(int odf, const char *op, int ndf, const char *np) {
    static renameat_fn_t real = NULL;
    if (!real) real = (renameat_fn_t)dlsym(RTLD_NEXT, "renameat");
    ensure_parent(np);
    return real(odf, op, ndf, np);
}

typedef int (*renameat2_fn_t)(int, const char *, int, const char *, unsigned int);
int renameat2(int odf, const char *op, int ndf, const char *np, unsigned int flags) {
    static renameat2_fn_t real = NULL;
    if (!real) real = (renameat2_fn_t)dlsym(RTLD_NEXT, "renameat2");
    ensure_parent(np);
    return real(odf, op, ndf, np, flags);
}
