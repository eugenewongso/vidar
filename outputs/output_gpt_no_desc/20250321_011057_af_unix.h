/* SPDX-License-Identifier: GPL-2.0 */
#ifndef __LINUX_NET_AFUNIX_H
#define __LINUX_NET_AFUNIX_H

#include <linux/socket.h>
#include <linux/un.h>
#include <linux/mutex.h>
#include <linux/refcount.h>
#include <net/sock.h>

void unix_inflight(struct user_struct *user, struct file *fp);
void unix_notinflight(struct user_struct *user, struct file *fp);
void unix_destruct_scm(struct sk_buff *skb);
void unix_gc(void);
void wait_for_unix_gc(void);
struct sock *unix_get_socket(struct file *filp);
struct sock *unix_peer_get(struct sock *sk);

#define UNIX_HASH_SIZE	256
#define UNIX_HASH_BITS	8

extern unsigned int unix_tot_inflight;
extern spinlock_t unix_table_lock;
extern struct hlist_head unix_socket_table[2 * UNIX_HASH_SIZE];

struct unix_address {
	refcount_t	refcnt;
	int		len;
	unsigned int	hash;
	struct sockaddr_un name[];
};

struct unix_skb_parms {
	struct pid		*pid;		/* Skb credentials	*/
	kuid_t			uid;
	kgid_t			gid;
	struct scm_fp_list	*fp;		/* Passed files		*/
#ifdef CONFIG_SECURITY_NETWORK
	u32			secid;		/* Security ID		*/
#endif
	u32			consumed;
} __randomize_layout;

struct scm_stat {
	atomic_t nr_fds;
};

#define UNIXCB(skb)	(*(struct unix_skb_parms *)&((skb)->cb))

#define unix_state_lock(s)	spin_lock(&unix_sk(s)->lock)
#define unix_state_unlock(s)	spin_unlock(&unix_sk(s)->lock)
#define unix_state_lock_nested(s) \
				spin_lock_nested(&unix_sk(s)->lock, \
				SINGLE_DEPTH_NESTING)

/* The AF_UNIX socket */
struct unix_sock {
	/* WARNING: sk has to be the first member */
	struct sock		sk;
	struct unix_address	*addr;
	struct path		path;
	struct mutex		iolock, bindlock;
	struct sock		*peer;
	struct list_head	link;
	unsigned long		inflight;
	spinlock_t		lock;
	unsigned long		gc_flags;
#define UNIX_GC_CANDIDATE	0
#define UNIX_GC_MAYBE_CYCLE	1
	struct socket_wq	peer_wq;
	wait_queue_entry_t	peer_wake;
	struct scm_stat		scm_stat;
};

static inline struct unix_sock *unix_sk(const struct sock *sk)
{
	return (struct unix_sock *)sk;
}

#define peer_wait peer_wq.wait

long unix_inq_len(struct sock *sk);
long unix_outq_len(struct sock *sk);

#ifdef CONFIG_SYSCTL
int unix_sysctl_register(struct net *net);
void unix_sysctl_unregister(struct net *net);
#else
static inline int unix_sysctl_register(struct net *net) { return 0; }
static inline void unix_sysctl_unregister(struct net *net) {}
#endif
#endif

static struct sock *unix_create1(struct net *net, struct socket *sock, int kern,
	int type)
{
	struct sock *sk;
	struct unix_sock *u;

	sk->sk_write_space	= unix_write_space;
	sk->sk_max_ack_backlog	= net->unx.sysctl_max_dgram_qlen;
	sk->sk_destruct		= unix_sock_destructor;
	u = unix_sk(sk);
	u->inflight = 0;
	u->path.dentry = NULL;
	u->path.mnt = NULL;
	spin_lock_init(&u->lock);
	INIT_LIST_HEAD(&u->link);
	mutex_init(&u->iolock); /* single task reading lock */
	mutex_init(&u->bindlock); /* single task binding lock */
	return sk;
}

static void dec_inflight(struct unix_sock *usk)
{
	usk->inflight--;
}

static void inc_inflight(struct unix_sock *usk)
{
	usk->inflight++;
}

static void inc_inflight_move_tail(struct unix_sock *u)
{
	u->inflight++;

	/* If this still might be part of a cycle, move it to the end
	 * of the list, so that it's checked even if it was already
	 * passed over
	 */
	/* rest of function body unchanged */
}

void unix_gc(void)
{
	/* ... previous code unchanged ... */
	list_for_each_entry_safe(u, next, &gc_inflight_list, link) {
		long total_refs;

		total_refs = file_count(u->sk.sk_socket->file);

		BUG_ON(!u->inflight);
		BUG_ON(total_refs < u->inflight);
		if (total_refs == u->inflight) {
			list_move_tail(&u->link, &gc_candidates);
			__set_bit(UNIX_GC_CANDIDATE, &u->gc_flags);
			__set_bit(UNIX_GC_MAYBE_CYCLE, &u->gc_flags);
		}
	}

	/* Move cursor to after the current position. */
	list_move(&cursor, &u->link);

	if (u->inflight) {
		list_move_tail(&u->link, &not_cycle_list);
		__clear_bit(UNIX_GC_MAYBE_CYCLE, &u->gc_flags);
		scan_children(&u->sk, inc_inflight_move_tail, NULL);
	}
	/* ... rest of function unchanged ... */
}

void unix_inflight(struct user_struct *user, struct file *fp)
{
	struct socket *s = sock_from_file(fp);

	if (s) {
		struct unix_sock *u = unix_sk(s);

		if (!u->inflight) {
			BUG_ON(!list_empty(&u->link));
			list_add_tail(&u->link, &gc_inflight_list);
		} else {
			BUG_ON(list_empty(&u->link));
		}
		u->inflight++;
		/* Paired with READ_ONCE() in wait_for_unix_gc() */
		WRITE_ONCE(unix_tot_inflight, unix_tot_inflight + 1);
	}
}

void unix_notinflight(struct user_struct *user, struct file *fp)
{
	struct socket *s = sock_from_file(fp);

	if (s) {
		struct unix_sock *u = unix_sk(s);

		BUG_ON(!u->inflight);
		BUG_ON(list_empty(&u->link));

		u->inflight--;
		if (!u->inflight)
			list_del_init(&u->link);
		/* Paired with READ_ONCE() in wait_for_unix_gc() */
		WRITE_ONCE(unix_tot_inflight, unix_tot_inflight - 1);
	}
}