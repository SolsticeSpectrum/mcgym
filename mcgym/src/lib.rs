//! mcgym: a fast headless Minecraft RL gym for the MCAI trainer.
//!
//! Built on Pumpkin's world simulation (vanilla-faithful), driven over shared
//! memory by the PyTorch PPO trainer. This crate currently exposes the schema
//! codec (the cross-language byte contract); the world/transport layers are
//! added on top of it.

pub mod gym;
pub mod obs;
pub mod physics;
pub mod registry;
pub mod schema;
pub mod transport;
pub mod world;
