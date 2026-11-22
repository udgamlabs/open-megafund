# Inputs of Fernandez, Stein and Lo (2012), Simulations A and B

- **Files:** `init_params_fn_NBTx_A.m` and `init_params_fn_NBTx_B.m`, copied unchanged from
  https://github.com/dfagnan/RBOMatlab, `Testing/`, at commit e43bb5758b954595a7f68fe7db782f4b13fe4aa0 (11 Jul 2015).
- **sha256:**
  - A: 6ef3ed4baa1e00691557d6c04b2e6d7d474e297d64a9e5f8aa12ce57f840ccd0
  - B: 5f5371b71dff3d0bafb711103446a3cf98e45438615e06bac03092328974ee73
- **Paper:** Fernandez, Stein and Lo, "Commercializing biomedical research through securitization techniques", Nature
  Biotechnology 30(10):964-975 (2012).
- **Code by** David Fagnan, Austin Gromatzky, Jose-Maria Fernandez, Roger M. Stein and Andrew W. Lo.
- **Licence:** the RBOMatlab repository's LICENSE file is MIT (2014, David Fagnan); its README and file headers say
  GPL-3.0-or-later. The files are kept here unchanged with their notices, and this repository is GPL-3.0-or-later, which
  satisfies both.
- **Pass marks:** `Testing/run_tests.m` in the same repository expects a mean annualised equity return of 0.091 (A) and
  0.106 (B). A run passes if it lands within 1.645 standard errors over 10,000 simulated funds.
