use doorman_shared::{Request, Response, AUTH_TIMEOUT_SECS, SOCKET_PATH};
use std::ffi::CStr;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

/// Connect to daemon and send authentication request
fn authenticate_user(username: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let socket = UnixStream::connect_timeout(
        &std::os::unix::net::SocketAddr::from_pathname(SOCKET_PATH)?,
        Duration::from_secs(AUTH_TIMEOUT_SECS),
    )?;
    
    socket.set_read_timeout(Some(Duration::from_secs(AUTH_TIMEOUT_SECS)))?;
    socket.set_write_timeout(Some(Duration::from_secs(1)))?;

    let request = Request::Authenticate {
        username: username.to_string(),
    };
    
    let mut stream = socket;
    let request_json = serde_json::to_string(&request)?;
    writeln!(stream, "{}", request_json)?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let mut response_line = String::new();
    reader.read_line(&mut response_line)?;

    let response: Response = serde_json::from_str(&response_line)?;

    match response {
        Response::Success { .. } => Ok(true),
        Response::Failure { .. } => Ok(false),
        _ => Ok(false),
    }
}

#[no_mangle]
pub extern "C" fn pam_sm_authenticate(
    pamh: *mut pam_sys::pam_handle_t,
    _flags: i32,
    _argc: i32,
    _argv: *const *const i8,
) -> i32 {
    unsafe {
        let mut user: *const i8 = std::ptr::null();
        let ret = pam_sys::pam_get_user(pamh, &mut user, std::ptr::null());
        
        if ret != pam_sys::PAM_SUCCESS as i32 {
            return pam_sys::PAM_AUTH_ERR as i32;
        }
        
        if user.is_null() {
            return pam_sys::PAM_USER_UNKNOWN as i32;
        }
        
        let username = match CStr::from_ptr(user).to_str() {
            Ok(s) => s,
            Err(_) => return pam_sys::PAM_AUTH_ERR as i32,
        };
        
        // Skip root
        if username == "root" {
            return pam_sys::PAM_AUTH_ERR as i32;
        }
        
        match authenticate_user(username) {
            Ok(true) => pam_sys::PAM_SUCCESS as i32,
            Ok(false) => pam_sys::PAM_AUTH_ERR as i32,
            Err(_) => pam_sys::PAM_AUTH_ERR as i32,
        }
    }
}

#[no_mangle]
pub extern "C" fn pam_sm_setcred(
    _pamh: *mut pam_sys::pam_handle_t,
    _flags: i32,
    _argc: i32,
    _argv: *const *const i8,
) -> i32 {
    pam_sys::PAM_SUCCESS as i32
}
