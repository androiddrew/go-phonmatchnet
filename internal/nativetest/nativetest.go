// Package nativetest runs native ONNX Runtime tests in fresh processes.
package nativetest

import (
	"bytes"
	"os"
	"os/exec"
	"regexp"
	"testing"
	"time"
)

const marker = "PHONMATCHNET_TEST_SUBPROCESS"

// Subprocess runs the calling test in a fresh process and reports whether the
// caller is that process. ortenv retains the ONNX Runtime environment until
// process exit, so a test that needs an uninitialized environment calls
//
//	if !nativetest.Subprocess(t) {
//		return
//	}
//
// after any skip checks and before touching ONNX Runtime.
func Subprocess(t *testing.T) bool {
	t.Helper()
	if os.Getenv(marker) == t.Name() {
		return true
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	args := []string{"-test.run=^" + regexp.QuoteMeta(t.Name()) + "$", "-test.count=1", "-test.v"}
	if deadline, ok := t.Deadline(); ok {
		args = append(args, "-test.timeout="+time.Until(deadline).String())
	}
	cmd := exec.CommandContext(t.Context(), executable, args...)
	cmd.Env = append(os.Environ(), marker+"="+t.Name())
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("native subprocess failed: %v\n%s", err, output)
	}
	// A child that matched no test or skipped also exits 0.
	if !bytes.Contains(output, []byte("--- PASS: "+t.Name()+" (")) {
		t.Fatalf("native subprocess did not pass %s:\n%s", t.Name(), output)
	}
	t.Logf("native subprocess:\n%s", output)
	return false
}
