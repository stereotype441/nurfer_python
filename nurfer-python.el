(defcustom nurfer-python-executable "python"
  "Pathname to the python executable"
  :type '(string))

(defcustom nurfer-undefined-names-script (concat (file-name-directory load-file-name) "undefined_names.py")
  "Pathname to the 'undefined_names.py' script"
  :type '(string))

(defun python-deperenthesize (expr) "If EXPR has enclosing parentheses, remove them."
  (if (and (string-match "\\`(" expr) (string-match ")\\'" expr))
      (substring expr 1 (match-beginning 0))
    expr))

(defun python-extract-variable (beg end varname) "Extract the text delimited by BEG and END, and replace it with a new variable named VARNAME."
  (interactive "r\nMVariable name: ")
  (let ((expr (buffer-substring-no-properties beg end)))
    (save-excursion
      (delete-region beg end)
      (goto-char beg)
      (insert varname)
      (python-beginning-of-statement)
      (let ((indent-level (current-indentation)))
	(save-excursion
	  (newline)
	  (indent-to indent-level)))
      (insert varname " = " (python-deperenthesize expr)))))

(global-set-key "\C-cpv" 'python-extract-variable)

(defun run-undefined-names-on-current-buffer (&rest options)
  "Run the undefined_names.py script, passing the given options to it."
  (let ((orig-buffer (current-buffer))
	(spans))
    (with-temp-buffer
      (let* ((temp-buffer (current-buffer))
	     (process-result
	      (save-excursion
		(set-buffer orig-buffer)
		(apply 'call-process-region
		       (point-min)
		       (point-max)
		       nurfer-python-executable
		       nil
		       temp-buffer
		       nil
		       nurfer-undefined-names-script
		       "-s"
		       options))))
	(if (not (= process-result 0))
	    (error "Problem running undefined_names.py")
	  (goto-char (point-min))
	  (read (current-buffer)))))))

(defun get-undefined-name-spans ()
  "Return a list of spans containing undefined names"
  (run-undefined-names-on-current-buffer))

(defun find-undefined-names ()
  "Run undefined_names.py to find names in the current buffer that are undefined."
  (interactive)
  (let ((spans (get-undefined-name-spans)))
    (remove-overlays (point-min) (point-max) 'nurfer 'undefined-name)
    (while spans
      (let* ((item (pop spans))
	     (overlay (make-overlay (1+ (car item)) (1+ (cadr item)))))
	(overlay-put overlay 'nurfer 'undefined-name)
	(overlay-put overlay 'face font-lock-warning-face)))))
(defun find-undefined-names-if-python ()
  (if (string= mode-name "Python")
      (find-undefined-names)))

(run-with-idle-timer 1 t 'find-undefined-names-if-python)

(defun compute-distance-to-span (span)
  (let ((span-start (1+ (nth 0 span)))
	(span-end (1+ (nth 1 span))))
    (cond
     ((< (point) span-start) (- span-start (point)))
     ((> (point) span-end) (- (point) span-end))
     (t 0))))

(defun apply-diff (diff)
  (let ((diff-start (1+ (nth 0 diff)))
	(diff-end (1+ (nth 1 diff)))
	(new-text (nth 2 diff)))
    (save-excursion
      (goto-char diff-start)
      (delete-region diff-start diff-end)
      (insert new-text))))

(defun do-nearest-quick-fix ()
  (interactive)
  (let ((spans (get-undefined-name-spans))
	(n 0)
	(nearest)
	(nearest-distance))
    (while spans
      (let* ((item (pop spans))
	     (distance (compute-distance-to-span item)))
	(if (or (null nearest-distance) (< distance nearest-distance))
	    (progn
	      (setq nearest-distance distance)
	      (setq nearest n)))
	(setq n (1+ n))))
    (if (null nearest)
	(error "No quick fixes available")
      (let ((annotated-span (car (run-undefined-names-on-current-buffer "-f" "-n" (prin1-to-string nearest)))))
	(apply-diff (cdr (nth 2 annotated-span)))))))

(global-set-key "\M-\C-m" (quote do-nearest-quick-fix))
