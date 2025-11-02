use doorman_shared::{Request, Response, AUTH_TIMEOUT_SECS, SOCKET_PATH};
use std::ffi::CStr;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

// PAM return codes (standard values)
const PAM_SUCCESS: libc::c_int = 0;
const PAM_AUTH_ERR: libc::c_int = 9;
const PAM_USER_UNKNOWN: libc::c_int = 10;

/// Connect to daemon and send authentication request
fn authenticate_user(username: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let socket = UnixStream::connect(SOCKET_PATH)?;
    
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
    pamh: *mut pam_sys::PamHandle,
    _flags: libc::c_int,
    _argc: libc::c_int,
    _argv: *const *const libc::c_char,
) -> libc::c_int {
    unsafe {
        let mut user: *const libc::c_char = std::ptr::null();
        let ret = pam_sys::raw::pam_get_user(pamh, &mut user, std::ptr::null());
        
        if ret != PAM_SUCCESS {
            return PAM_AUTH_ERR;
        }
        
        if user.is_null() {
            return PAM_USER_UNKNOWN;
        }
        
        let username = match CStr::from_ptr(user).to_str() {
            Ok(s) => s,
            Err(_) => return PAM_AUTH_ERR,
        };
        
        // Skip root
        if username == "root" {
            return PAM_AUTH_ERR;
        }
        
        match authenticate_user(username) {
            Ok(true) => PAM_SUCCESS,
            Ok(false) => PAM_AUTH_ERR,
            Err(_) => PAM_AUTH_ERR,
        }
    }
}

#[no_mangle]
pub extern "C" fn pam_sm_setcred(
    _pamh: *mut pam_sys::PamHandle,
    _flags: libc::c_int,
    _argc: libc::c_int,
    _argv: *const *const libc::c_char,
) -> libc::c_int {
    PAM_SUCCESS
}
