//! Packed ScheduleFreeAdamW: training holds y, evaluation holds the average x.
use crate::{Result, engine::Engine};
use kernite_cuda::Grid;
use serde_json::{Value, json};

#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f32, p: u64, i: u32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        ld.global.f32(v, [a]);
    }
    fn store(p: u64, i: u32, v: f32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        st.global.f32([a], v);
    }
    fn index(i: u32) {
        let (b, t, n): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
    }
    pub unsafe extern "ptx-kernel" fn switch(
        p: *mut f32,
        base: *const f32,
        count: u32,
        weight: f32,
    ) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (y, z, delta): (f32, f32, f32);
        load(y, p, i);
        load(z, base, i);
        sub.rn.f32(delta, z, y);
        mul.rn.f32(delta, delta, weight);
        add.rn.f32(y, y, delta);
        store(p, i, y);
    }
    pub unsafe extern "ptx-kernel" fn step(
        p: *mut f32,
        base: *mut f32,
        second: *mut f32,
        gradients: *const f32,
        count: u32,
        rate: f32,
        average_weight: f32,
        correction: f32,
        query_coefficient: f32,
    ) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (y, z, v, g, u, delta, den): (f32, f32, f32, f32, f32, f32, f32);
        load(y, p, i);
        load(z, base, i);
        load(v, second, i);
        load(g, gradients, i);
        mul.rn.f32(v, v, 0.98);
        mul.rn.f32(u, g, g);
        mul.rn.f32(u, u, 0.02);
        add.rn.f32(v, v, u);
        div.rn.f32(den, v, correction);
        sqrt.rn.f32(den, den);
        add.rn.f32(den, den, 1e-8);
        div.rn.f32(u, g, den);
        // Historical decay=1, applied at the pre-update query y.
        add.rn.f32(u, u, y);
        sub.rn.f32(delta, z, y);
        mul.rn.f32(delta, delta, average_weight);
        add.rn.f32(y, y, delta);
        mul.rn.f32(delta, u, query_coefficient);
        add.rn.f32(y, y, delta);
        mul.rn.f32(delta, u, rate);
        sub.rn.f32(z, z, delta);
        store(p, i, y);
        store(base, i, z);
        store(second, i, v);
    }
}

pub struct Optimizer {
    kernels: kernels::Kernels,
    pub completed: usize,
    maximum: f64,
    total: f64,
    pub training: bool,
}
impl Optimizer {
    pub fn new(e: &mut Engine) -> Result<Self> {
        e.first_moment.write(&e.parameters.read()?)?;
        e.second_moment.write(&vec![0.; e.parameters.len()])?;
        Ok(Self {
            kernels: kernels::Kernels::load(&e.cuda)?,
            completed: 0,
            maximum: 0.,
            total: 0.,
            training: true,
        })
    }
    pub fn restore(e: &Engine, state: &Value) -> Result<Self> {
        Ok(Self {
            kernels: kernels::Kernels::load(&e.cuda)?,
            completed: state["completed"]
                .as_u64()
                .ok_or("missing completed")?
                .try_into()?,
            maximum: state["maximum"].as_f64().ok_or("missing maximum")?,
            total: state["total"].as_f64().ok_or("missing total")?,
            training: state["training"]
                .as_bool()
                .ok_or("missing optimizer mode")?,
        })
    }
    pub fn state(&self) -> Value {
        json!({"completed":self.completed,"maximum":self.maximum,"total":self.total,"training":self.training})
    }
    pub fn mode(&mut self, e: &Engine, training: bool) -> Result<()> {
        if self.training != training {
            // SAFETY: all buffers have the validated parameter count; one writer per entry.
            unsafe {
                self.kernels.switch(
                    Grid::new(e.parameters.len().div_ceil(256) as u32, 256),
                    e.parameters.ptr(),
                    e.first_moment.ptr(),
                    e.parameters.len() as u32,
                    if training {
                        0.1
                    } else {
                        (1. - 1. / 0.9) as f32
                    },
                )?;
            }
            self.training = training;
        }
        Ok(())
    }
    pub fn step(&mut self, e: &Engine) -> Result<()> {
        if !self.training {
            return Err("ScheduleFree step requires query y".into());
        }
        let t = (self.completed + 1) as f64;
        let rate = 0.03 * (t / 10.).min(1.);
        self.maximum = self.maximum.max(rate);
        self.total += self.maximum * self.maximum;
        let c = self.maximum * self.maximum / self.total;
        // SAFETY: validated parameter arrays have matching lengths and ordered launches.
        unsafe {
            self.kernels.step(
                Grid::new(e.parameters.len().div_ceil(256) as u32, 256),
                e.parameters.ptr(),
                e.first_moment.ptr(),
                e.second_moment.ptr(),
                e.gradients.ptr(),
                e.parameters.len() as u32,
                rate as f32,
                c as f32,
                (1. - 0.98_f64.powf(t)) as f32,
                (rate * (0.9 * (1. - c) - 1.)) as f32,
            )?;
        }
        self.completed += 1;
        Ok(())
    }
}
