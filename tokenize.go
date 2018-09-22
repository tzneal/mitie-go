package mitie

/*
#cgo LDFLAGS: -lmitie

#include <stdlib.h>

#include "mitie.h"
*/
import "C"
import (
	"unsafe"
)

// Tokenize tokenizes a string in the same manner as is used by model training..
func Tokenize(s string) []string {
	cs := C.CString(s)
	defer C.free(unsafe.Pointer(cs))
	ret := C.mitie_tokenize(cs)
	if ret == nil {
		return nil
	}
	defer C.mitie_free(unsafe.Pointer(ret))

	tokens := []string{}
	p := (*[1 << 30]*C.char)(unsafe.Pointer(ret))
	i := 0
	for p[i] != nil {
		tokens = append(tokens, C.GoString(p[i]))
		i++
	}
	return tokens
}

//  TokenizeWithOffsets tokenizes the string the same as Tokenizes,
//  but also returns an array of offsets within the string s, to the
//  beginning of each token.
func TokenizeWithOffsets(s string) ([]string, []int) {
	cs := C.CString(s)
	defer C.free(unsafe.Pointer(cs))

	var offs *C.ulong
	v := C.mitie_tokenize_with_offsets(cs, &offs)
	if v == nil {
		return nil, nil
	}
	defer C.mitie_free(unsafe.Pointer(v))
	defer C.mitie_free(unsafe.Pointer(offs))

	p := (*[1 << 30]*C.char)(unsafe.Pointer(v))
	o := (*[1 << 30]C.ulong)(unsafe.Pointer(offs))
	i := 0
	tokens := []string{}
	offsets := []int{}
	for p[i] != nil {
		tokens = append(tokens, C.GoString(p[i]))
		offsets = append(offsets, int(o[i]))
		i++
	}

	return tokens, offsets
}
