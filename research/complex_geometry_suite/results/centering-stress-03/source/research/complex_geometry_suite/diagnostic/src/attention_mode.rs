use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttentionMode {
    Xsa,
    Ordinary,
    Centered,
}

impl AttentionMode {
    pub fn from_ordinary(ordinary: bool) -> Self {
        if ordinary { Self::Ordinary } else { Self::Xsa }
    }
    pub fn tag(self) -> &'static str {
        match self {
            Self::Xsa => "hermitian_xsa",
            Self::Ordinary => "ordinary_causal",
            Self::Centered => "centered_causal",
        }
    }
    pub fn from_protocol(protocol: &Value) -> Result<Self> {
        match protocol["attention"].as_str() {
            None | Some("hermitian_xsa") => Ok(Self::Xsa),
            Some("ordinary_causal") => Ok(Self::Ordinary),
            Some("centered_causal") => Ok(Self::Centered),
            _ => Err("unknown checkpoint attention".into()),
        }
    }
    pub fn enable(self, engine: &mut Engine) -> Result<()> {
        match self {
            Self::Xsa => Ok(()),
            Self::Ordinary => engine.enable_ordinary_attention(),
            Self::Centered => engine.enable_centered_attention(),
        }
    }
    pub fn control_key(self) -> &'static str {
        match self {
            Self::Xsa => "radial_control_files",
            Self::Ordinary => "plain_xsa_control_files",
            Self::Centered => "ordinary_control_files",
        }
    }
}
