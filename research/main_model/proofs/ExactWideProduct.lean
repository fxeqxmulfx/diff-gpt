import Transformer.Precision.ArithmeticError

/-!
Exact finite FP32 products through integer significands, 2026-11-08.
The construction follows integer_product_native/src/kernels.rs: 24x24-bit
multiplication, bit-length normalization and IEEE binary64 field assembly.
The theorem certifies this arithmetic construction; it does not verify the
CUDA compiler. Native tests compare the generated kernel independently.
-/
open Transformer.Precision
namespace MainModelExactProduct

def significand32 (e m : ℕ) : ℕ := if e = 0 then m else 8388608 + m
def exponent32 (e : ℕ) : ℤ := if e = 0 then -149 else (e : ℤ) - 150
def sign (negative : Bool) : ℝ := if negative then -1 else 1

theorem significand32_range (e m : ℕ) (hm : m < 8388608) :
    significand32 e m < 16777216 := by
  unfold significand32
  split <;> omega

theorem exponent32_range (e : ℕ) (he : e < 255) :
    -149 ≤ exponent32 e ∧ exponent32 e ≤ 104 := by
  unfold exponent32
  split <;> omega

theorem binary32_fields (negative : Bool) (e m : ℕ) :
    binary32Value negative e m =
      sign negative * (significand32 e m : ℝ) * (2 : ℝ) ^ exponent32 e := by
  unfold binary32Value significand32 exponent32 sign
  split_ifs <;> push_cast <;> ring

/-- Direct assembly of normal binary64 fields decodes to the stated value. -/
theorem binary64_normal_decode (negative : Bool) (e m : ℕ)
    (he0 : 0 < e) (he : e < 2047) (hm : m < 4503599627370496) :
    let bits := (if negative then 9223372036854775808 else 0) +
      4503599627370496 * e + m
    bits < 18446744073709551616 ∧
    ieee 11 52 bits = some (sign negative * (4503599627370496 + (m : ℝ)) *
      (2 : ℝ)^((e : ℤ)-1075)) := by
  intro bits
  have hb : bits < 18446744073709551616 := by
    dsimp [bits]; split <;> omega
  have hf : bits % 4503599627370496 = m := by
    dsimp [bits]; split <;> omega
  have hx : bits / 4503599627370496 % 2048 = e := by
    dsimp [bits]; split <;> omega
  have hs : bits / 9223372036854775808 % 2 = if negative then 1 else 0 := by
    dsimp [bits]; split <;> omega
  refine ⟨hb, ?_⟩
  norm_num only [ieee, bias]
  rw [hf, hx, hs]
  have he' : e ≠ 2047 := by omega
  have hz : e ≠ 0 := by omega
  have hexp : (e : ℤ)-1023-52 = (e : ℤ)-1075 := by ring
  cases negative <;> simp [sign, he', hz, hexp]

def encodeDyadic (negative : Bool) (p : ℕ) (exponent : ℤ) : ℕ :=
  (if negative then 9223372036854775808 else 0) +
  if p = 0 then 0 else
    4503599627370496 * (exponent + (p.log2 : ℤ) + 1023).toNat +
      (p * 2^(52-p.log2) - 4503599627370496)

/-- Every at-most48-bit significand at an FP32-product exponent is encoded
exactly in a normal binary64, or as signed zero. There is no rounding premise. -/
theorem encodeDyadic_exact (negative : Bool) (p : ℕ) (exponent : ℤ)
    (hp : p < 281474976710656) (he : -298 ≤ exponent ∧ exponent ≤ 208) :
    encodeDyadic negative p exponent < 18446744073709551616 ∧
    ieee 11 52 (encodeDyadic negative p exponent) =
      some (sign negative * (p : ℝ) * (2 : ℝ)^exponent) := by
  by_cases hz : p = 0
  · subst p
    cases negative <;> norm_num [encodeDyadic, sign, ieee, bias]
  let k := p.log2
  have hk : k ≤ 47 := by
    have h := (Nat.log2_lt hz).2 (show p < 2^48 by norm_num; exact hp)
    dsimp [k]; omega
  have hlo : 2^k ≤ p := Nat.log2_self_le hz
  have hhi : p < 2^(k+1) := Nat.lt_log2_self
  let q := p * 2^(52-k)
  have powlo : (2 : ℕ)^k * 2^(52-k) = 2^52 := by
    rw [← pow_add]; congr 1; omega
  have powhi : (2 : ℕ)^(k+1) * 2^(52-k) = 2^53 := by
    rw [← pow_add]; congr 1; omega
  have hqlo : 4503599627370496 ≤ q := by
    have h := Nat.mul_le_mul_right (2^(52-k)) hlo
    rw [powlo] at h
    exact h
  have hqhi : q < 9007199254740992 := by
    have h := Nat.mul_lt_mul_of_pos_right hhi (by positivity : 0 < (2 : ℕ)^(52-k))
    rw [powhi] at h
    exact h
  let m := q - 4503599627370496
  let e := (exponent + (k : ℤ) + 1023).toNat
  have heq : (e : ℤ) = exponent + (k : ℤ) + 1023 :=
    Int.toNat_of_nonneg (by omega)
  have he0 : 0 < e := by omega
  have hel : e < 2047 := by omega
  have hm : m < 4503599627370496 := by dsimp [m]; omega
  have hq : (4503599627370496 : ℝ) + (m : ℝ) =
      (p : ℝ) * (2 : ℝ)^(52-k) := by
    have h : 4503599627370496 + m = q := by dsimp [m]; omega
    have hr : (4503599627370496 : ℝ) + (m : ℝ) = (q : ℝ) := by exact_mod_cast h
    rw [hr]; dsimp [q]; push_cast; rfl
  have hpowers : (2 : ℝ)^(52-k) * 2^((e : ℤ)-1075) = 2^exponent := by
    have hc : (2 : ℝ)^(52-k) = (2 : ℝ)^((52-k : ℕ) : ℤ) := by simp
    rw [hc, ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    congr 1
    omega
  have h := binary64_normal_decode negative e m he0 hel hm
  have hbits : encodeDyadic negative p exponent =
      (if negative then 9223372036854775808 else 0) + 4503599627370496 * e + m := by
    simp [encodeDyadic, hz, e, m, q, k, Nat.add_assoc]
  rw [hbits]
  refine ⟨h.1, ?_⟩
  rw [h.2, hq]
  congr 1
  calc
    sign negative * ((p : ℝ) * 2^(52-k)) * 2^((e : ℤ)-1075) =
        sign negative * p * (2^(52-k) * 2^((e : ℤ)-1075)) := by ring
    _ = _ := by rw [hpowers]

/-- The constructed bits encode the exact product of ANY two finite IEEE
binary32 inputs, including subnormals. Every product uses at most48bits and
has a binary64 normal exponent; arbitrary raw-gradient scale is covered. -/
theorem finite_binary32_product_exact (sa sb : Bool) (ea ma eb mb : ℕ)
    (hea : ea < 255) (hma : ma < 8388608) (heb : eb < 255) (hmb : mb < 8388608) :
    let p := significand32 ea ma * significand32 eb mb
    let exponent := exponent32 ea + exponent32 eb
    encodeDyadic (sa != sb) p exponent < 18446744073709551616 ∧
    ieee 11 52 (encodeDyadic (sa != sb) p exponent) =
      some (binary32Value sa ea ma * binary32Value sb eb mb) := by
  intro p exponent
  have ha := significand32_range ea ma hma
  have hb := significand32_range eb mb hmb
  have hp : p < 281474976710656 := by
    dsimp [p]
    have h := Nat.mul_self_lt_mul_self (show max (significand32 ea ma) (significand32 eb mb) < 16777216 by omega)
    nlinarith [Nat.le_max_left (significand32 ea ma) (significand32 eb mb),
      Nat.le_max_right (significand32 ea ma) (significand32 eb mb)]
  have he : -298 ≤ exponent ∧ exponent ≤ 208 := by
    have ha := exponent32_range ea hea
    have hb := exponent32_range eb heb
    dsimp [exponent]; omega
  have h := encodeDyadic_exact (sa != sb) p exponent hp he
  refine ⟨h.1, ?_⟩
  rw [h.2, binary32_fields, binary32_fields]
  congr 1
  dsimp [p, exponent]
  rw [zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
  push_cast
  cases sa <;> cases sb <;> norm_num [sign] <;> ring

example := finite_binary32_product_exact false true 187 0 207 0
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example := finite_binary32_product_exact false false 0 1 0 1
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)

#print axioms encodeDyadic_exact
#print axioms finite_binary32_product_exact
end MainModelExactProduct
