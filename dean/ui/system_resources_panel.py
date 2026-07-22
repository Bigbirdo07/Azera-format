from __future__ import annotations

import platform

import streamlit as st

from nlp.system_resources import (
    available_memory_mb,
    list_top_processes,
    ollama_call_is_safe,
    quit_process_gracefully,
    swap_percent,
)


def render_system_resources_panel() -> None:
    with st.expander("System resources (free memory for the AI)"):
        free_mb = available_memory_mb()
        swap = swap_percent()
        safe, reason = ollama_call_is_safe()

        if free_mb is None:
            st.caption("Memory stats aren't available on this system.")
        else:
            st.caption(f"Free memory: {free_mb:.0f} MB · Swap used: {swap:.0f}%"
                       if swap is not None else f"Free memory: {free_mb:.0f} MB")
            if safe:
                st.success("AI: available")
            else:
                st.warning(f"AI: running on rules only — {reason}")

        # st.rerun() below starts a fresh render immediately, so a
        # success/error shown right before it would never actually be seen.
        # Stash the result and show it on the render that follows instead.
        last_result = st.session_state.pop("_quit_result", None)
        if last_result:
            ok, message = last_result
            (st.success if ok else st.error)(message)

        if platform.system() != "Darwin":
            st.caption("Closing other apps from here is only supported on macOS right now.")
            return

        processes = list_top_processes()
        if not processes:
            st.caption("No process information available.")
            return

        st.caption("Top memory-consuming apps. Quitting is graceful (same as "
                   "quitting normally) and asks you to confirm first.")
        pending_pid = st.session_state.get("_pending_quit_pid")

        for row in processes:
            pid, name, mem_mb = row["pid"], row["name"], row["mem_mb"]
            st.write(f"**{name}** — {mem_mb:.0f} MB")

            if pending_pid == pid:
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("Confirm", key=f"quit_confirm_{pid}", use_container_width=True):
                    ok, error = quit_process_gracefully(pid, name)
                    st.session_state["_pending_quit_pid"] = None
                    st.session_state["_quit_result"] = (
                        (True, f"Asked {name} to quit.") if ok else (False, error)
                    )
                    st.rerun()
                if cancel_col.button("Cancel", key=f"quit_cancel_{pid}", use_container_width=True):
                    st.session_state["_pending_quit_pid"] = None
                    st.rerun()
            else:
                if st.button("Quit…", key=f"quit_proc_{pid}", use_container_width=True):
                    st.session_state["_pending_quit_pid"] = pid
                    st.rerun()
