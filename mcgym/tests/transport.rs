//! End-to-end transport test: a Rust stand-in for the Python `ShmTransport` client
//! drives a trivial gym over the real shm file + UDS, exercising the header,
//! RESET/STEP/CLOSE handshake, and the action->obs data path.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::thread;
use std::time::Duration;

use memmap2::Mmap;
use mcgym::schema::{ACTION_NBYTES, Action, OBS_NBYTES, Obs};
use mcgym::transport::{
    CMD_CLOSE, CMD_RESET, CMD_STEP, HEADER_NBYTES, MAGIC, REPLY_OK, Transport, Gym,
};

const N: usize = 3;

/// Stamps each agent's obs with its index + the tick, and echoes that agent's
/// commanded `forward` into `health` so the client can confirm action delivery.
struct EchoGym {
    tick: i64,
}

impl EchoGym {
    fn write(&self, actions: Option<&[u8]>, obs: &mut [u8]) {
        for i in 0..N {
            let mut o = Obs {
                agent_id: i as i32,
                tick: self.tick,
                ..Default::default()
            };
            if let Some(a) = actions {
                let act = Action::decode(&a[i * ACTION_NBYTES..(i + 1) * ACTION_NBYTES]);
                o.health = act.forward;
            }
            o.encode_into(&mut obs[i * OBS_NBYTES..(i + 1) * OBS_NBYTES]);
        }
    }
}

impl Gym for EchoGym {
    fn reset(&mut self, obs: &mut [u8]) {
        self.tick = 0;
        self.write(None, obs);
    }
    fn step(&mut self, actions: &[u8], obs: &mut [u8]) {
        self.tick += 1;
        self.write(Some(actions), obs);
    }
}

#[test]
fn transport_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let shm = dir.path().join("shm.bin");
    let sock = dir.path().join("gym.sock");

    let (shm_t, sock_t) = (shm.clone(), sock.clone());
    let server = thread::spawn(move || {
        let mut t = Transport::create(&shm_t, &sock_t, N).unwrap();
        let mut gym = EchoGym { tick: -1 };
        t.accept().unwrap();
        t.serve(&mut gym).unwrap();
    });

    // Client: connect (retry until the server has bound), then attach to shm.
    let mut stream = loop {
        if let Ok(s) = UnixStream::connect(&sock) {
            break s;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let file = std::fs::File::open(&shm).unwrap();
    let map = unsafe { Mmap::map(&file).unwrap() };

    // Header checks (mirror ShmTransport.__init__).
    assert_eq!(i32::from_le_bytes(map[0..4].try_into().unwrap()), MAGIC);
    assert_eq!(i32::from_le_bytes(map[4..8].try_into().unwrap()), 1);
    assert_eq!(i32::from_le_bytes(map[8..12].try_into().unwrap()), N as i32);

    let obs_off = HEADER_NBYTES + N * ACTION_NBYTES;
    let obs_at = |i: usize| -> Obs {
        let base = obs_off + i * OBS_NBYTES;
        Obs::decode(&map[base..base + OBS_NBYTES])
    };
    let send = |stream: &mut UnixStream, cmd: u8| {
        stream.write_all(&[cmd]).unwrap();
        let mut reply = [0u8; 1];
        stream.read_exact(&mut reply).unwrap();
        assert_eq!(reply[0], REPLY_OK);
    };

    // RESET -> tick 0, agent ids stamped.
    send(&mut stream, CMD_RESET);
    for i in 0..N {
        let o = obs_at(i);
        assert_eq!(o.agent_id, i as i32);
        assert_eq!(o.tick, 0);
    }

    // Write actions into the shm ACTION region, STEP, confirm echo + tick advance.
    {
        let mut wfile = std::fs::OpenOptions::new().read(true).write(true).open(&shm).unwrap();
        use std::io::{Seek, SeekFrom};
        for i in 0..N {
            let act = Action { forward: 1.5 + i as f32, ..Default::default() };
            wfile.seek(SeekFrom::Start((HEADER_NBYTES + i * ACTION_NBYTES) as u64)).unwrap();
            wfile.write_all(&act.encode()).unwrap();
        }
        wfile.flush().unwrap();
    }
    send(&mut stream, CMD_STEP);
    for i in 0..N {
        let o = obs_at(i);
        assert_eq!(o.tick, 1, "tick advanced");
        assert_eq!(o.health, 1.5 + i as f32, "action forward echoed into health");
    }

    send(&mut stream, CMD_CLOSE);
    server.join().unwrap();
}
