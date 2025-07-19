// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded affix "><li class="part-title">Fundamentals</li><li class="chapter-item expanded "><a href="chapter_001.html"><strong aria-hidden="true">1.</strong> Learning Rate and Optimization</a></li><li class="chapter-item expanded "><a href="chapter_002.html"><strong aria-hidden="true">2.</strong> Train-Test Split Strategies</a></li><li class="chapter-item expanded "><a href="chapter_003.html"><strong aria-hidden="true">3.</strong> Covariance vs Correlation</a></li><li class="chapter-item expanded "><a href="chapter_004.html"><strong aria-hidden="true">4.</strong> Skewed Distributions</a></li><li class="chapter-item expanded "><a href="chapter_005.html"><strong aria-hidden="true">5.</strong> Loss Function Robustness: MAE vs MSE vs RMSE</a></li><li class="chapter-item expanded affix "><li class="part-title">Recommendation Systems</li><li class="chapter-item expanded "><a href="chapter_006.html"><strong aria-hidden="true">6.</strong> Content-Based vs Collaborative Filtering</a></li><li class="chapter-item expanded "><a href="chapter_007.html"><strong aria-hidden="true">7.</strong> Restaurant Recommendation Systems</a></li><li class="chapter-item expanded affix "><li class="part-title">Advanced Machine Learning</li><li class="chapter-item expanded "><a href="chapter_008.html"><strong aria-hidden="true">8.</strong> Ensemble Methods and Performance</a></li><li class="chapter-item expanded "><a href="chapter_009.html"><strong aria-hidden="true">9.</strong> Focal Loss in Object Detection</a></li><li class="chapter-item expanded "><a href="chapter_010.html"><strong aria-hidden="true">10.</strong> Clock Hands Angle Problem</a></li><li class="chapter-item expanded affix "><li class="part-title">Data Optimization and Few-Shot Learning</li><li class="chapter-item expanded "><a href="chapter_011.html"><strong aria-hidden="true">11.</strong> Optimizing Labeled Data</a></li><li class="chapter-item expanded "><a href="chapter_012.html"><strong aria-hidden="true">12.</strong> Few-Shot Learning</a></li><li class="chapter-item expanded "><a href="chapter_013.html"><strong aria-hidden="true">13.</strong> Greedy Layer-wise Pretraining</a></li><li class="chapter-item expanded "><a href="chapter_014.html"><strong aria-hidden="true">14.</strong> Freezing Transformer Layers</a></li><li class="chapter-item expanded "><a href="chapter_015.html"><strong aria-hidden="true">15.</strong> Dropout During Inference</a></li><li class="chapter-item expanded affix "><li class="part-title">Generative Models</li><li class="chapter-item expanded "><a href="chapter_016.html"><strong aria-hidden="true">16.</strong> Variational Autoencoders</a></li><li class="chapter-item expanded "><a href="chapter_017.html"><strong aria-hidden="true">17.</strong> Generative Models: Training vs Inference</a></li><li class="chapter-item expanded "><a href="chapter_018.html"><strong aria-hidden="true">18.</strong> Subword Tokenization</a></li><li class="chapter-item expanded "><a href="chapter_019.html"><strong aria-hidden="true">19.</strong> Sigmoid for Numerical Prediction</a></li><li class="chapter-item expanded "><a href="chapter_020.html"><strong aria-hidden="true">20.</strong> Function Derivative Zero Sum</a></li><li class="chapter-item expanded affix "><li class="part-title">More chapters will be added as we generate them...</li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
