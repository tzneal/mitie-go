package mitie

/*
#cgo LDFLAGS: -lmitie

#include <stdlib.h>

#include "mitie.h"
*/
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

type RelationDetector struct {
	x *C.mitie_binary_relation_detector
}

func NewRelationDetector(filename string) (*RelationDetector, error) {
	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))

	rd := C.mitie_load_binary_relation_detector(cs)
	if rd == nil {
		return nil, errors.New("unable to load relation detector")
	}
	r := &RelationDetector{rd}
	runtime.SetFinalizer(r, freeRelationDetector)
	return r, nil
}

func freeRelationDetector(r *RelationDetector) {
	C.mitie_free(unsafe.Pointer(r.x))
}

func (r RelationDetector) Name() string {
	// no need to free here, it's valid until the RelationDetector
	// itself is free'd
	name := C.mitie_binary_relation_detector_name_string(r.x)
	return C.GoString(name)
}
